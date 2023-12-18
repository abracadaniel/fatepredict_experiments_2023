from copy import deepcopy
import logging
import os
import time

import daisy
import pylp

from linajea.utils import CandidateDatabase
from .daisy_check_functions import (  # noqa: F401
    check_function, write_done,
    check_function_all_blocks,
    write_done_all_blocks)
from linajea.tracking import track, greedy_track


from linajea_cost_test import get_edge_indicator_costs, get_node_indicator_costs, get_constraints



logger = logging.getLogger(__name__)




def solve_blockwise(linajea_config):
    """Function to solve an ILP-based optimization problem block-wise
    Notes
    -----
    The set of constraints has been predefined.
    For each block:
    Takes the previously object candidates and extracted edge
    candidates, sets up the objective based on their score and creates
    the respective constraints for all indicator variables and solve.
    To achieve consistent solutions along the block boundary, compute
    with overlap, blocks without overlap can be processed in parallel.
    If there is overlap, compute these blocks sequentially; if a
    solution for candidates in the overlap area has already been
    computed by a previous block, enforce this in the remaining blocks.
    Args
    ----
    linajea_config: TrackingConfig
        Configuration object
    """
    parameters = deepcopy(linajea_config.solve.parameters)
    _verify_parameters(parameters)
    # block_size is identical for all parameters

    # block_size = [ 2, 512, 512, 712,]

    block_size = daisy.Coordinate(parameters[0].block_size)

    # context = [ 2, 100, 100, 100,]
    context = daisy.Coordinate(linajea_config.solve.context)

    data = linajea_config.inference_data.data_source
    db_name = data.db_name
    db_host = linajea_config.general.db_host
    solve_roi = daisy.Roi(offset=data.roi.offset,
                          shape=data.roi.shape)

    # determine parameters id from database
    graph_provider = CandidateDatabase(
        db_name,
        db_host)
    
    '''
    Get id for parameter set from mongo collection.
    If fail_if_not_exists, fail if the parameter set isn't already there.
    The default is to assign a new id and write it to the collection.
    '''
    parameters_id = [graph_provider.get_parameters_id(p) for p in parameters]

    if linajea_config.solve.from_scratch:
        graph_provider.reset_selection(parameter_ids=parameters_id)
        if len(parameters_id) > 1:
            graph_provider.database.drop_collection(
                'solve_' + str(hash(frozenset(parameters_id))) + '_daisy')

    if linajea_config.solve.greedy:
        block_write_roi = solve_roi
    else:
        block_write_roi = daisy.Roi(
            (0, 0, 0, 0),
            block_size)
    block_read_roi = block_write_roi.grow(
        context,
        context)
    total_roi = solve_roi.grow(
        context,
        context)

    logger.info("Solving in %s", total_roi)
    if data.datafile is not None:
        logger.info("Sample: %s", data.datafile.filename)
    logger.info("DB: %s", db_name)

    param_names = ['solve_' + str(_id) for _id in parameters_id]
    if len(parameters_id) > 1:
        # check if set of parameters is already done
        step_name = 'solve_' + str(hash(frozenset(parameters_id)))
        if check_function_all_blocks(step_name, db_name, db_host):
            logger.info("Param set with name %s already completed. Exiting",
                        step_name)
            return parameters_id
    else:
        step_name = 'solve_' + str(parameters_id[0])
    # Check each individual parameter to see if it is done
    # if it is, remove it from the list
    done_indices = []
    for _id, name in zip(parameters_id, param_names):
        if check_function_all_blocks(name, db_name, db_host):
            logger.info("Params with id %d already completed. Removing", _id)
            done_indices.append(parameters_id.index(_id))
    for index in done_indices[::-1]:
        del parameters_id[index]
        del parameters[index]
        del param_names[index]
    logger.debug(parameters_id)
    if len(parameters_id) == 0:
        logger.info("All parameters in set already completed. Exiting")
        return parameters_id

    task = daisy.Task(
        "linajea_solving",
        total_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: solve_in_block(
            linajea_config,
            parameters_id,
            b,
            solution_roi=solve_roi),
        # Note: in the case of a set of parameters,
        # we are assuming that none of the individual parameters are
        # half done and only checking the hash for each block
        check_function=lambda b: check_function(
            b,
            step_name,
            db_name,
            db_host),
        num_workers=linajea_config.solve.job.num_workers,
        fit='overhang')

    success = daisy.run_blockwise([task])
    # TODO: Temporarily disabled due to bug in daisy, server returns True
    # even if some blocks failed, as long as they have been processed.
    # if success:
    #     # write all done to individual parameters and set
    #     if len(param_names) > 1:
    #         write_done_all_blocks(
    #             step_name,
    #             db_name,
    #             db_host)
    #     for name in param_names:
    #         write_done_all_blocks(
    #             name,
    #             db_name,
    #             db_host)
    logger.info("Finished solving")
    return parameters_id if success else success









def _verify_parameters(parameters):
    block_size = parameters[0].block_size
    for i in range(len(parameters)):
        assert block_size == parameters[i].block_size, \
            "%s not equal to %s" %\
            (block_size, parameters[i].block_size)
        #assert parameters[i].max_cell_move is not None, \
        #    f"max_cell_move has to be set for parameter set {parameters[i]}"


def solve_in_block(linajea_config,
                   parameters_id,
                   block,
                   solution_roi=None):

    # Solution_roi is the total roi that you want a solution in
    # Limiting the block to the solution_roi allows you to solve
    # all the way to the edge, without worrying about reading
    # data from outside the solution roi
    # or paying the appear or disappear costs unnecessarily


    db_name = linajea_config.inference_data.data_source.db_name
    db_host = linajea_config.general.db_host

    if len(parameters_id) == 1:
        step_name = 'solve_' + str(parameters_id[0])
    else:
        _id = hash(frozenset(parameters_id))
        step_name = 'solve_' + str(_id)
    logger.info("Solving in block %s", block)

    if solution_roi:
        # Limit block to source_roi
        logger.debug("Block write roi: %s", block.write_roi)
        logger.debug("Solution roi: %s", solution_roi)
        read_roi = block.read_roi.intersect(solution_roi)
        write_roi = block.write_roi.intersect(solution_roi)
    else:
        read_roi = block.read_roi
        write_roi = block.write_roi

    logger.debug("Write roi: %s", str(write_roi))
    
    if write_roi.empty:
        logger.info("Write roi empty, skipping block %d", block.block_id)
        write_done(block, step_name, db_name, db_host)
        return 0
    
    graph_provider = CandidateDatabase(
        db_name,
        db_host,
        mode='r+')


    parameters = graph_provider.get_parameters(parameters_id[0])
    start_time = time.time()
    selected_keys = ['selected_' + str(pid) for pid in parameters_id]
    edge_attrs = selected_keys.copy()
    edge_attrs.extend(["prediction_distance", "distance"])
    join_collection = parameters.get("cell_state_key")
    logger.info("join collection %s", join_collection)
    graph = graph_provider.get_graph(
            read_roi,
            edge_attrs=edge_attrs,
            join_collection=join_collection
            )

    dangling_nodes = [
        n
        for n, data in graph.nodes(data=True)
        if 't' not in data
    ]
    graph.remove_nodes_from(dangling_nodes)


    if linajea_config.solve.clip_low_score:
        logger.info("Dropping low score nodes")
        low_score_nodes = [
            n
            for n, data in graph.nodes(data=True)
            #Todo what socre
            if data['score'] < linajea_config.solve.clip_low_score
        ]
        graph.remove_nodes_from(low_score_nodes)

    
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    logger.info("Reading graph with %d nodes and %d edges took %s seconds",
                num_nodes, num_edges, time.time() - start_time)
    

    constrs = get_constraints()
    solver = track(graph, linajea_config,
                    selected_keys,
                    frame_key='t',
                    edge_indicator_costs=get_edge_indicator_costs,
                    node_indicator_costs=get_node_indicator_costs,
                    constraints_fns=constrs[0],
                    pin_constraints_fns=constrs[1]
                    )