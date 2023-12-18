import zarr
import daisy
import numpy as np
import time
import networkx as nx
from linajea_cost_test import get_merge_graph_from_array
from funlib.math import encode64, decode64
import linajea
from linajea_cost_test import get_edge_indicator_costs, get_node_indicator_costs, get_constraints


import time
import logging

logger = logging.getLogger(__name__)

class SolvingConfig():
    def __init__(self, solve_config):
        self.solve = solve_config


class TrackingConfig():
    def __init__(self, solve_config):
        self.solve = solve_config

def overlap(frg_pre, frg_next):
    f_pre = frg_pre.flatten()
    f_next = frg_next.flatten()
    pairs, counts = np.unique(np.stack([f_pre, f_next]), axis=1,
                              return_counts=True)
    return pairs.T, counts

# find root and leaves and connect edge


def provide_root(graph):
    roots = [x for x in graph.nodes() if graph.in_degree(x) == 0]
    return roots

def provide_leaves(graph):
    leaves = [x for x in graph.nodes() if graph.out_degree(x) == 0]
    return leaves

def iterate_tree(graph,r):
    iter_list = [r,*list(nx.dfs_predecessors(graph,r))]
    iter_list.reverse()
    return iter_list

def connect_edge(A,B,merge_tree_pre,merge_tree_nex,ids_pre,ids_nex,pairs,counts):
    '''
    
    provide all predecessors of node r 

    Args
    ---
    A,B: graph node id
    
    merge_tree_pre,merge_tree_nex : nx.DiGraph()

    ids_pre,ids_nex : ndarray from zarr file Fragment_stats/id/pre and Fragment_stats/id/nex

    pairs : list from func overlap 
    
    counts : list from func overlap

    '''
    
    count=0
    sub_a = iterate_tree(merge_tree_pre,A)
    sub_b = iterate_tree(merge_tree_nex,B)
    A_area = decode64(int(A),dims=5,bits=[9,12,12,12,19])[4]
    B_area = decode64(int(B),dims=5,bits=[9,12,12,12,19])[4]
    for a in sub_a:
        for b in sub_b:
            if a in list(ids_pre) and b in list(ids_nex):
                label_a = list(ids_pre).index(a)+1
                label_b = list(ids_nex).index(b)+1
                ind = np.where(np.all(pairs == [[label_a,label_b]], axis=1))
                if len(ind[0]>0):
                    #a,b is id will not be show in pairs
                    index = ind[0].item()
                    count += counts[index]

    return count/(A_area + B_area - count) #iou

def add_nodes_from_merge_tree(G1,G2):
    ''''
    G1: merge graph
    G2: candidate graph
    '''
    for node in G1.nodes():
        # Add the node to G2, copying over the attributes
        G2.add_node(node, **G1.nodes[node])
        # Check if the node has any incoming edges in G1
        parent = node
        # if no edges in node the parent is its self
        for edge in G1.in_edges(node):
                parent = edge[0]
        # update the node with `parent`
        G2.nodes[node]['parent'] = parent
        G2.nodes[node]['id'] = node
    return G2

def create_candidate_graph(file_name):
    z = zarr.open ( file_name, 'r' )
    fragments = z['Fragments'][:]
    print(z.tree(level=3))
    graph_fragments=nx.DiGraph()
    start_time = time.time()
    for t in range(fragments.shape[0]-1):
        print('frame: %s / %s' % (t, fragments.shape[0]-1))
        pre = t 
        nex = t + 1
        z = zarr.open(file_name,'r')
        ids_pre = z['Fragment_stats/id/'+str(pre)]
        ids_nex = z['Fragment_stats/id/'+str(nex)]
        merge_tree_pre = z['Merge_tree/Merge/'+str(pre)]
        merge_tree_nex = z['Merge_tree/Merge/'+str(nex)]
        scores_pre = z['Merge_tree/Scoring/'+str(pre)]
        scores_nex = z['Merge_tree/Scoring/'+str(nex)]

        merge_tree_pre = get_merge_graph_from_array(merge_tree_pre,scores_pre)
        print('-merged pre')
        merge_tree_nex = get_merge_graph_from_array(merge_tree_nex,scores_nex)
        print('-merged next')


        graph_fragments = add_nodes_from_merge_tree(merge_tree_pre,graph_fragments)
        print('-added nodes pre')
        graph_fragments = add_nodes_from_merge_tree(merge_tree_nex,graph_fragments)
        print('-added nodes next')

        # find overlaping pairs
        pairs, counts = overlap (fragments[pre], fragments[nex]) 
        # pairs is label not the ID!!!!
           
        # iterate tree to extract edge
        
        root = provide_root(merge_tree_pre)
        # set_oder to iterate merge_tree
        iter_list_A = iterate_tree(merge_tree_pre,root[0])
        root = provide_root(merge_tree_nex)
        iter_list_B = iterate_tree(merge_tree_nex,root[0])
        # iterate two merge_tree and connect new edges
        print('-iterating over A&B')
        for c, A in enumerate(iter_list_A):
            print('--iterating A: %s / %s' % (c, len(iter_list_A)))
            for B in iter_list_B: 
                # create edges for connecting the node a in merge tree merge_pre and node a in merge tree merge_nex
                count = connect_edge(A,B,merge_tree_pre,merge_tree_nex,ids_pre,ids_nex,pairs,counts)
                # add edegs
                if count != 0:
                    graph_fragments.add_edge(B, A, source = B, target = A, overlap = count)
                    #print('add',a,b , 'count',count)

        print(" iterating one merge tree cost: ", time.time() - start_time)
    
    #nx.write_gexf(graph_fragments, "anno_alice_T2030_candidategrah.gexf")
    print('The candidate graph was saved and it cost:',time.time() - start_time)
    return(graph_fragments)


if __name__ == "__main__":
    
    file_name = "/groups/funke/home/xuz2/anno_alice_T2030.zarr"
    z = zarr.open ( file_name, 'r' )
    fragments = z['Fragments'][:]
    raw = z['Raw'][:]
    #track = z['gt_trackimage']
    print(z.tree(level=2))
    
    # fragments_edge
    #graph_fragments = nx.DiGraph()
    candidate_edge = {}
    graph_fragments=nx.DiGraph()

    # set a timer
    start_time = time.time()

    for t in range(fragments.shape[0]-1):
        pre = t 
        nex = t + 1
        z = zarr.open(file_name,'r')
        ids_pre = z['Fragment_stats/id/'+str(pre)]
        ids_nex = z['Fragment_stats/id/'+str(nex)]
        merge_tree_pre = z['Merge_tree/Merge/'+str(pre)]
        merge_tree_nex = z['Merge_tree/Merge/'+str(nex)]
        scores_pre = z['Merge_tree/Scoring/'+str(pre)]
        scores_nex = z['Merge_tree/Scoring/'+str(nex)]

        merge_tree_pre = get_merge_graph_from_array(merge_tree_pre,scores_pre)
        merge_tree_nex = get_merge_graph_from_array(merge_tree_nex,scores_nex)


        graph_fragments = add_nodes_from_merge_tree(merge_tree_pre,graph_fragments)
        graph_fragments = add_nodes_from_merge_tree(merge_tree_nex,graph_fragments)

        # find overlaping pairs
        pairs, counts = overlap (fragments[pre], fragments[nex]) 
        # pairs is label not the ID!!!!

        '''
        for (label_pre, label_nex), count in zip(pairs, counts):
            id_pre = ids_pre[int(label_pre-1)]
            id_nex = ids_nex[int(label_nex-1)]
            graph_fragments.add_edge(id_nex, id_pre, source = id_nex, target = id_pre, overlap = count)

        '''
        
        
        # iterate tree to extract edge
        
        root = provide_root(merge_tree_pre)
        # set_oder to iterate merge_tree
        iter_list_A = iterate_tree(merge_tree_pre,root[0])
        root = provide_root(merge_tree_nex)
        iter_list_B = iterate_tree(merge_tree_nex,root[0])
        # iterate two merge_tree and connect new edges
        for a in iter_list_A:
            sub_a = iterate_tree(merge_tree_pre,a)
            for b in iter_list_B: 
                sub_b = iterate_tree(merge_tree_nex,b)
                # create edges for connecting the node a in merge tree merge_pre and node a in merge tree merge_nex
                count = connect_edge(a,b,merge_tree_pre,merge_tree_nex,ids_pre,ids_nex,pairs,counts)
                # add edegs
                if count != 0:
                    graph_fragments.add_edge(b, a, source = b, target = a, overlap = count)
                    print('add',a,b , 'count',count)

                    print(" iterating time: ", time.time() - start_time)
    
    nx.write_gexf(graph_fragments, "anno_alice_T2030_candidategrah.gexf")
    print('The candidate graph was saved and it cost:',time.time() - start_time)
        
    
    
          
    
    # 3 graph

    # put to solver
    
    roi = daisy.Roi((0, 0, 0, 0), (3, 15, 25, 25))

    # input graph
    graph = linajea.tracking.TrackGraph(graph_fragments, frame_key='t', roi=roi)
    for u, v, data in graph.edges(data=True):
        if graph.nodes[u]['score'] != 0.0001:
                print(u,'has socre',graph.nodes[u]['score'])

    
    #print(graph.edges(data=True))

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
    solve_config.timeout = 100000
    


#solve_config.solver_type = None

    config = TrackingConfig(solve_config)
    constrs = get_constraints()

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

    
    #save solution graph
    G = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        #print(u,v,data)
        if data['selected']:
            G.add_node(u, **graph.nodes[u])
            G.add_node(v, **graph.nodes[v])
            G.add_edge(u,v,**data)
            if graph.nodes[u]['score'] != 0:
                print(u,'has socre',graph.nodes[u]['score'])
    nx.write_gexf(G, "solution_graph/Alice_anno_solution.gexf")

    # print result
    selected_edges = []
    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append((u, v))
        if data['selected']:
            selected_edges.append((u, v))

    print('the number of selected edges: ', len(selected_edges))
    print('selected edges:', selected_edges)


    #check soluthon
    #print(G.nodes(data=True))
        
