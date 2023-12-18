import numpy as np
import networkx as nx
import daisy
import pylp
import linajea.config
import linajea.tracking
import linajea.tracking.track
import linajea.tracking.cost_functions
import linajea.utils
from linajea.tracking.cost_functions import is_nth_frame
from linajea.tracking.cost_functions import is_close_to_roi_border
from linajea.tracking.constraints import ensure_edge_endpoints, ensure_at_most_two_successors, ensure_one_predecessor, ensure_pinned_edge, ensure_split_set_for_divs
from funlib.math import encode64, decode64

class TrackingConfig():
    def __init__(self, solve_config):
        self.solve = solve_config


def feature_times_weight_costs_fn(weight, key="score",
                                  feature_func=lambda x: x):
    """ pass feature times weight to solver object
    Args
    -----
    weight: float
    key: string, feature name
    feature_func: callable function
    """
    def fn(obj):
        feature = feature_func(obj[key])
        return feature, weight

    return fn


def get_edge_indicator_costs(parameters, graph):
    """Get a predefined map of edge indicator costs functions
    Args
    ----
    parameters: SolveParametersConfig
        Current set of weights and parameters used to compute costs.
    graph: TrackGraph
        Graph containing the node candidates for which the costs will be
        computed (not used for the default edge costs).
    """
    fn_map = {
        "edge_selected": [
            feature_times_weight_costs_fn(parameters.weight_edge_score,
                                          key="overlap",
                                          feature_func=lambda x: x)]
    }

    return fn_map


def constant_costs_fn(weight, zero_if_true=lambda _: False):

    def fn(obj):
        feature = 1
        cond_weight = 0 if zero_if_true(obj) else weight
        return feature, cond_weight

    return fn


def get_node_indicator_costs(parameters, graph):
    """Get a predefined map of node indicator costs functions
    Args
    ----
    parameters: SolveParametersConfig
        Current set of weights and parameters used to compute costs.
    graph: TrackGraph
        Graph containing the node candidates for which the costs will be
        computed.
    """

    feature_func = lambda x: x

    fn_map = {
        "node_selected": [
            feature_times_weight_costs_fn(
                parameters.weight_node_score,
                key="score", feature_func=feature_func), #change the key = 'size' for set the score
            constant_costs_fn(parameters.selection_constant)],
        "node_appear": [
            constant_costs_fn(
                parameters.track_cost,
                zero_if_true=lambda obj: (
                    is_nth_frame(graph.begin)(obj) or
                     is_close_to_roi_border(
                         graph.roi, parameters.max_cell_move)(obj)))],
        "node_split":  [
            constant_costs_fn(1)]
    }

    return fn_map


def get_merge_graph_from_array(merge_tree, scores):
    """Get graph representation from array representation of the merge tree
    merge_tree: ndarray
        (n, 3) array where n is the number of merges and each item holds
        indices (u, v, w): u and v merge to form w.
        We assume (and therefore do not check) that the indices are unique and
        there are no loops.
    scores: ndarray
        (n,) array defining the merge score for each of the merges in the merge
        tree
    get merge tree from zarr array
    """
    G = nx.DiGraph()

    for (u, v, w), score in zip(merge_tree, scores):
        pos = decode64(int(w), dims=5, bits=[9,12,12,12,19])
        G.add_node(w, t =pos[0], z=pos[1], y=pos[2], x=pos[3], score=score, size = pos[4])
        G.add_edge(w, u)
        G.add_edge(w, v)

    # set leave node attributes
    for node in G.nodes:
        if 'score' not in G.nodes[node]:
            pos = decode64(int(node), dims=5, bits=[9,12,12,12,19])
            nx.set_node_attributes(G, {node: {'t': pos[0], 'z': pos[1], 'y': pos[2], 'x': pos[3], 'score': 0.0001, 'size': pos[4]}}) # set the fragments with a small value

    return G


def get_merge_graph(candidate_graph, t):
    """Get merge graph from Linajea CandidateGraph
    The nodes can be chosen for that time point.
    The edges can be obtained from the "parent" attribute of the node.
    Parameters
    ----------
    candidate_graph: linajea.CandidateGraph
        The Candidate graph containing all the nodes of interest
    t: int
        The time to consider
    """
    g = nx.DiGraph()
    nodes = [nid for nid in candidate_graph.nodes()
                if candidate_graph.nodes[nid]['t'] == t]
    # Iteratively add nodes
    for nid in nodes:
        node = candidate_graph.nodes[nid]
        if nid not in g:
            g.add_node(nid,**candidate_graph.nodes[nid])

        if node['parent'] != nid and node['id']:   
        # determine a root as node['parent'] = nid in candidate_graph func add_nodes_from_merge_tree()
            g.add_edge(nid, node['parent'])
    return g


def get_conflict_sets(graph):
    """Get conflict sets from merge tree.
    Nodes are in conflict if they are along the same path.
    ----------
    graph: nx.Graph
    """
    # Get all leaves - no incoming edges
    leaves = [x for x in graph.nodes() if graph.in_degree(x) == 0]
    # Get all roots - no outgoing edges
    roots = [x for x in graph.nodes() if graph.out_degree(x) == 0]
    # Get all paths from a leaf to a root
    conflict_sets = []
    for root in roots:
        for leaf in leaves:
            for path in nx.all_simple_paths(graph, source=leaf, target=root):
                conflict_sets.append(path)
    return conflict_sets


def set_conflict_sets(graph, indicators):
    """Certain sets of nodes are mutually exclusive, e.g. they cannot be chosen
    at the same time.
    For example: if we choose a merged node, we cannot at the same time choose
    one of the fragments from which it was made.
    Constraint:
        sum( selected(node) for node in conflict_set ) <= 1
    Parameters
    ----------
    graph: TrackGraph
        Graph containing the node and edge candidates, the ILP will be solved
        for this graph.
    indicators: dict str: dict (int or pair of int): int
        Contains a dict for every indicator type (str).
        Each dict maps from node (int) or edge (pair of int) candidate to
        the corresponding indicator variable/index (int). Each candidate can
        have indicators of different types associated to it.
    """
    time = []
    constraints = []
    for cell in graph.nodes:
        t = graph.nodes[cell]['t']
        time.append(t)
    for t in np.unique(time):
        merge_graph = get_merge_graph(graph, t)
        conflict_sets = get_conflict_sets(merge_graph)
        for cs in conflict_sets:
            constraint = pylp.LinearConstraint()

            for node in cs:
                constraint.set_coefficient(indicators["node_selected"][node], 1)

            # Relation, value
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(1)
            constraints.append(constraint)

    return constraints


def get_constraints():
    pin_constraints_fn_list = [ensure_pinned_edge]
    constraints_fn_list = [ensure_edge_endpoints,
                           ensure_one_predecessor,
                           ensure_at_most_two_successors,
                           ensure_split_set_for_divs,
                           set_conflict_sets]
    return (constraints_fn_list,
            pin_constraints_fn_list)


if __name__ == "__main__":
    # test demo
    cells = [
        {'id': 1, 't': 0, 'z': 1, 'y': 1, 'x': 2, 'score': 0.4, 'parent': 1},
        {'id': 2, 't': 0, 'z': 1, 'y': 1, 'x': 1, 'score': 0.2, 'parent': 1},
        {'id': 3, 't': 0, 'z': 1, 'y': 1, 'x': 3, 'score': 0.2, 'parent': 1},
        {'id': 4, 't': 0, 'z': 1, 'y': 1, 'x': 0, 'score': 0, 'parent': 2},
        {'id': 5, 't': 0, 'z': 1, 'y': 1, 'x': 2, 'score': 0, 'parent': 2}
    ]

    graph1 = nx.DiGraph()
    graph1.add_nodes_from([(cell['id'], cell) for cell in cells])

    cells = [
        {'id': 6, 't': 1, 'z': 1, 'y': 1, 'x': 3, 'score': 0.4, 'parent': 6},
        {'id': 7, 't': 1, 'z': 1, 'y': 1, 'x': 1, 'score': 1.0, 'parent': 6},
        {'id': 8, 't': 1, 'z': 1, 'y': 1, 'x': 3, 'score': 1.0, 'parent': 6},
        {'id': 9, 't': 1, 'z': 1, 'y': 1, 'x': 4, 'score': 1.0, 'parent': 6},
        {'id': 10, 't': 1, 'z': 1, 'y': 1, 'x': 0, 'score': 2.0, 'parent': 7},
        {'id': 11, 't': 1, 'z': 1, 'y': 1, 'x': 2, 'score': 2.0, 'parent': 7}
    ]

    graph2 = nx.DiGraph()
    graph2.add_nodes_from([(cell['id'], cell) for cell in cells])
    edges = [
            {'source': 6, 'target': 1, 'overlap': 1},
            {'source': 7, 'target': 1, 'overlap': 0.33},
            {'source': 8, 'target': 1, 'overlap': 0.33},
            {'source': 9, 'target': 1, 'overlap': 0.33},
            {'source': 10, 'target': 1, 'overlap': 0.16},
            {'source': 11, 'target': 1, 'overlap': 0.16},

            {'source': 6, 'target': 2, 'overlap': 0.5},
            {'source': 7, 'target': 2, 'overlap': 0.66},
            {'source': 8, 'target': 2, 'overlap': 0.5},
            {'source': 9, 'target': 2, 'overlap': 0},
            {'source': 10, 'target': 2, 'overlap': 0.33},
            {'source': 11, 'target': 2, 'overlap': 0.33},

            {'source': 6, 'target': 3, 'overlap': 0.5},
            {'source': 7, 'target': 3, 'overlap': 0.01},
            {'source': 8, 'target': 3, 'overlap': 0.5},
            {'source': 9, 'target': 3, 'overlap': 0.9},
            {'source': 10, 'target': 3, 'overlap': 0.001},
            {'source': 11, 'target': 3, 'overlap': 0.0},

            {'source': 6, 'target': 4, 'overlap': 0.33},
            {'source': 7, 'target': 4, 'overlap': 0.8},
            {'source': 8, 'target': 4, 'overlap': 0.01},
            {'source': 9, 'target': 4, 'overlap': 0.02},
            {'source': 10, 'target': 4, 'overlap': 0.5},
            {'source': 11, 'target': 4, 'overlap': 0.5},

            {'source': 6, 'target': 5, 'overlap': 0.3},
            {'source': 7, 'target': 5, 'overlap': 0.1},
            {'source': 8, 'target': 5, 'overlap': 0.75},
            {'source': 9, 'target': 5, 'overlap': 0.001},
            {'source': 10, 'target': 5, 'overlap': 0.02},
            {'source': 11, 'target': 5, 'overlap': 0.03},

    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(graph1.nodes(data=True))
    graph.add_nodes_from(graph2.nodes(data=True))
    graph.add_edges_from([(edge['source'], edge['target'], edge)
                         for edge in edges])

    roi = daisy.Roi((0, 0, 0, 0), (4, 5, 5, 5))
    
    # input graph
    graph = linajea.tracking.TrackGraph(graph, frame_key='t', roi=roi)

    ps = {
        "track_cost": 4.0,
        "weight_edge_score": -0.1,
        "weight_node_score": -0.1,
        "selection_constant": -1.0,
        "max_cell_move": 0.0,
        "block_size": [5, 100, 100, 100],
        }
    job = {"num_workers": 5, "queue": "normal"}

    solve_config = linajea.config.SolveConfig(parameters=ps,
                                              job=job,
                                              context=[2, 100, 100, 100])

    solve_config.solver_type = None
    solve_config.timeout = 1000
    config = TrackingConfig(solve_config)

    # TODO get_constarints from config
    constrs = get_constraints()
    graph = linajea.tracking.TrackGraph(graph, frame_key='t', roi=roi)
    linajea.tracking.track(
                graph,
                config,
                frame_key='t',
                selected_key='selected',
                edge_indicator_costs=get_edge_indicator_costs,
                node_indicator_costs=get_node_indicator_costs,
                constraints_fns=constrs[0],
                pin_constraints_fns=constrs[1]
                )

    # print result
    selected_edges = []
    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append((u, v))
        if data['selected']:
            selected_edges.append((u, v))

    print('all edges graph:', edges)
    print('selected edges:', selected_edges)

