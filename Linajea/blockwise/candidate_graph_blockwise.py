import logging
import time
import zarr
import numpy as np
import daisy
from .daisy_check_functions import write_done, check_function
import linajea
import networkx as nx
from linajea_cost_test import get_merge_graph_from_array
import time
from .coordinate import Coordinate
from .roi import Roi
from .freezable import Freezable
from .array import Array

logger = logging.getLogger(__name__)


def extract_edges_blockwise(linajea_config):

    """
    Function to extract edges (compute edge candidates between
    neighboring object candidates in adjacent frames)
    Starts a number of worker processes to process the blocks.
    Args
    ----
    linajea_config: TrackingConfig
        Configuration object
    """
    data = linajea_config.inference_data.data_source
    extract_roi = daisy.Roi(offset=data.roi.offset,
                            shape=data.roi.shape)
    

    

    assert linajea_config.solve.context, (
        "config.solve.context is not set, cannot determine how much spatial"
        " context is necessary for tracking (cells moving in and out of a"
        " block at the boundary). Please set it!")
    

    # set roi


    # block size in world units
    block_write_roi = daisy.Roi(
        (0,)*4,
        daisy.Coordinate(linajea_config.extract.block_size))

    # set thresgold
    max_edge_move_th = max(linajea_config.extract.edge_move_threshold.values())
    pos_context = daisy.Coordinate((0,) + (max_edge_move_th,)*3)
    neg_context = daisy.Coordinate((1,) + (max_edge_move_th,)*3)


    input_roi = extract_roi
    block_read_roi = block_write_roi

    logger.info("Following ROIs in world units:")
    logger.info("Input ROI       = %s", input_roi)
    logger.info("Block read  ROI = %s", block_read_roi)
    logger.info("Block write ROI = %s", block_write_roi)
    logger.info("Output ROI      = %s", extract_roi)

    logger.info("Starting block-wise processing...")


    
    

    container = daisy.open_ds(
                        data.datafile.filename,
                        'Fragments'
                        )
    

    def extract_edges_in_block(linajea_config, block):


        graph_provider = linajea.utils.CandidateDatabase(data.db_name,
                                                         linajea_config.general.db_host,
                                                          mode='r+')
        
        graph = graph_provider[block.read_roi]

        fragments = container[block.read_roi].to_ndarray()
        
        t_begin = block.read_roi.begin[0]
        t_end = block.read_roi.end[0]
        for t in range(t_begin,t_end-1):
            pre = t 
            nex = t + 1

            z = zarr.open(data.datafile.filename,'r')
            ids_pre = z['Fragment_stats/id/'+str(pre)]
            ids_nex = z['Fragment_stats/id/'+str(nex)]
            merge_tree_pre = z['Merge_tree/Merge/'+str(pre)]
            merge_tree_nex = z['Merge_tree/Merge/'+str(nex)]
            scores_pre = z['Merge_tree/Scoring/'+str(pre)]
            scores_nex = z['Merge_tree/Scoring/'+str(nex)]

            merge_tree_pre = get_merge_graph_from_array(merge_tree_pre,scores_pre)
            merge_tree_nex = get_merge_graph_from_array(merge_tree_nex,scores_nex)

            '''
            graph_fragments = add_nodes_from_merge_tree(merge_tree_pre,graph_fragments)
            graph_fragments = add_nodes_from_merge_tree(merge_tree_nex,graph_fragments)
            '''

            pairs, counts = overlap (fragments[pre], fragments[nex])

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
                    count = connect_edge(sub_a,sub_b,ids_pre,ids_nex,pairs,counts)
                    # add edegs
                    if count != 0:
                        graph.add_edge(b, a, source = b, target = a, overlap = count)
                        #graph.add_node_from(merge_tree_pre[a])
                        #graph.add_node_from(merge_tree_nex[b])
            
            graph.write_edges(block.write_roi)

        return 0




    task = daisy.Task(
        "cabdidategraph_extract_edges",
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: extract_edges_in_block(
            linajea_config,
            b),
        num_workers=linajea_config.extract.job.num_workers,
        read_write_conflict=False,
        fit='overhang')

    daisy.run_blockwise([task])





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

def connect_edge(sub_a,sub_b,ids_pre,ids_nex,pairs,score):
    # create edges for connecting the node a in merge tree merge_pre and node a in merge tree merge_nex
    count=0
    for a in sub_a:
        for b in sub_b:
            if a in list(ids_pre) and b in list(ids_nex):
                label_a = list(ids_pre).index(a)+1
                label_b = list(ids_nex).index(b)+1
                ind = np.where(np.all(pairs == [[label_a,label_b]], axis=1))
                if len(ind[0]>0):
                    #a,b is id will not be show in pairs
                    index = ind[0].item()
                    count += score[index] 
    return count

def add_nodes_from_merge_tree(G1,G2):
    for node in G1.nodes():
            # Add the node to G2, copying over the attributes
            G2.add_node(node, **G1.nodes[node])
            # Check if the node has any incoming edges in G1
            parent = None
            for edge in G1.in_edges(node):
                    parent = edge[0]
            # update the node with `parent`
            G2.nodes[node]['parent'] = parent
            G2.nodes[node]['id'] = node
    return G2


def open_ds(filename: str, ds_name: str, mode: str = "r"):
    """Open a Zarr, N5, or HDF5 dataset as an :class:`Array`. If the
    dataset has attributes ``resolution`` and ``offset``, those will be
    used to determine the meta-information of the returned array.
    Args:
        filename:
            The name of the container "file" (which is a directory for Zarr and
            N5).
        ds_name:
            The name of the dataset to open.
    Returns:
        A :class:`Array` pointing to the dataset.
    """

    if filename.endswith(".zarr") or filename.endswith(".zip"):
        assert (
            not filename.endswith(".zip") or mode == "r"
        ), "Only reading supported for zarr ZipStore"

        logger.debug("opening zarr dataset %s in %s", ds_name, filename)
        try:
            ds = zarr.open(filename, mode=mode)[ds_name]
        except Exception as e:
            logger.error("failed to open %s/%s" % (filename, ds_name))
            raise e

        voxel_size, offset = _read_voxel_size_offset(ds, ds.order)
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened zarr dataset %s in %s", ds_name, filename)
        return Array(ds, roi, voxel_size, chunk_shape=chunk_shape)

    elif filename.endswith(".n5"):
        logger.debug("opening N5 dataset %s in %s", ds_name, filename)
        ds = zarr.open(filename, mode=mode)[ds_name]

        voxel_size, offset = _read_voxel_size_offset(ds, "F")
        shape = Coordinate(ds.shape[-len(voxel_size) :])
        roi = Roi(offset, voxel_size * shape)

        chunk_shape = ds.chunks

        logger.debug("opened N5 dataset %s in %s", ds_name, filename)
        return Array(ds, roi, voxel_size, chunk_shape=chunk_shape)



    else:
        logger.error("don't know data format of %s in %s", ds_name, filename)
        raise RuntimeError("Unknown file format for %s" % filename)


def _read_voxel_size_offset(ds, order="C"):
    voxel_size = None
    offset = None
    dims = None

    if "resolution" in ds.attrs:
        voxel_size = Coordinate(ds.attrs["resolution"])
        dims = len(voxel_size)
    elif "scale" in ds.attrs:
        voxel_size = Coordinate(ds.attrs["scale"])
        dims = len(voxel_size)
    elif "pixelResolution" in ds.attrs:
        voxel_size = Coordinate(ds.attrs["pixelResolution"]["dimensions"])
        dims = len(voxel_size)

    elif "transform" in ds.attrs:
        # Davis saves transforms in C order regardless of underlying
        # memory format (i.e. n5 or zarr). May be explicitly provided
        # as transform.ordering
        transform_order = ds.attrs["transform"].get("ordering", "C")
        voxel_size = Coordinate(ds.attrs["transform"]["scale"])
        if transform_order != order:
            voxel_size = Coordinate(voxel_size[::-1])
        dims = len(voxel_size)

    if "offset" in ds.attrs:
        offset = Coordinate(ds.attrs["offset"])
        if dims is not None:
            assert dims == len(
                offset
            ), "resolution and offset attributes differ in length"
        else:
            dims = len(offset)

    elif "transform" in ds.attrs:
        transform_order = ds.attrs["transform"].get("ordering", "C")
        offset = Coordinate(ds.attrs["transform"]["translate"])
        if transform_order != order:
            offset = Coordinate(offset[::-1])

    if dims is None:
        dims = len(ds.shape)

    if voxel_size is None:
        voxel_size = Coordinate((1,) * dims)

    if offset is None:
        offset = Coordinate((0,) * dims)

    if order == "F":
        offset = Coordinate(offset[::-1])
        voxel_size = Coordinate(voxel_size[::-1])

    if voxel_size is not None and (offset / voxel_size) * voxel_size != offset:
        # offset is not a multiple of voxel_size. This is often due to someone defining
        # offset to the point source of each array element i.e. the center of the rendered
        # voxel, vs the offset to the corner of the voxel.
        # apparently this can be a heated discussion. See here for arguments against
        # the convention we are using: http://alvyray.com/Memos/CG/Microsoft/6_pixel.pdf
        logger.debug(
            f"Offset: {offset} being rounded to nearest voxel size: {voxel_size}"
        )
        offset = (
            (Coordinate(offset) + (Coordinate(voxel_size) / 2)) / Coordinate(voxel_size)
        ) * Coordinate(voxel_size)
        logger.debug(f"Rounded offset: {offset}")

    return Coordinate(voxel_size), Coordinate(offset)


