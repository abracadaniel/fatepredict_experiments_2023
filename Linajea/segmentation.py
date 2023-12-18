# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:07:34 2022

@author: wayne
"""

import configargparse
import cv2
import mahotas as mh
import numpy as np
import waterz
import zarr
from funlib.math import encode64
from skimage.measure import regionprops


from write_to_db import create_cells_client

def segment_stats(fragments, t):
    regions = regionprops(fragments)
    position = []
    region_size = []
    idx = []
    for props in regions:
        z0, y0, x0 = props.centroid
        position.append((t, int(z0), int(y0), int(x0)))
        ids = encode64((t, int(z0), int(y0), int(x0), int(props.area)),bits=[9,12,12,12,19])
        idx.append(ids)
        region_size.append(props.area)
        
    return np.array(idx), np.array(position), np.array(region_size)

def merge_stats(fragments, t):
    regions = regionprops(fragments)

    z0, y0, x0 = regions[0].centroid
    position = ((t, int(z0), int(y0), int(x0)))
    # TODO add bits for shifting.
    ids = encode64((t, int(z0), int(y0), int(x0), int(regions[0].area)),bits=[9,12,12,12,19])

    return ids, position, regions[0].area


def get_fragments(image):
    """Apply watershed to an image to get (over-)segmentation fragments.

    Parameters
    ----------
    image: array
        Boundary prediction image.
    """
    # normalized to 255 can get better watershed output
    inputimage = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # watershed algorithm set local minima as seed
    minima = mh.regmin(inputimage)
    markers, nr_markers = mh.label(minima)
    fragments = mh.cwatershed(inputimage, markers, return_lines=False)
    fragments = fragments.astype('uint64')
    return fragments


def get_affinities(image):
    """Get affinities from boundary predictions."""
    # normalized to 0 - 1 get affinity graph
    # TODO Might not be necessary, since data seems to be 0 - 1 already
    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    # affinity graph needs 3 channel input, original is just 1
    affs = np.zeros((3,) + image.shape)
    pre_nor = NormalizeData(image)

    # Invert the picture set membranes with low affinity
    # and cells with high affinity
    for i in range(3):
        affs[i] = pre_nor*-1+1

    # make sure correct type
    aff = affs.astype('float32')
    return aff

def segmentation_process(z, cells=[], deploy=False, max_volume=25000):
    # input image is the 4th channel image of data
    # for example our data is (c,t,z,y,x)
    # Usually normalized 0 - 1
    channel, time, *_ = z['Raw'].shape
    frag_t = []
    for t in range(time):
        image = z['Raw'][3, t, :, :, :]
        print(image.shape)
        affs = get_affinities(image)
        fragments = get_fragments(image)
        thresholds = [0, 100]
        frag_t.append(get_fragments(image))
        gen = waterz.agglomerate(affs, thresholds, fragments=fragments,
                                 return_merge_history=True,
                                 return_region_graph=True)
        # the initial seg is the fragments, with threshold=0
        segs, _, _ = next(gen)
        # prevent generator change data in RAM 
        seg = segs.copy()
        print('labels:', len(np.unique(seg)))
        # Fragment values in segs starts at 1
        ids, positions, volumes = segment_stats(fragments, t)
        z['Fragment_stats/id/'+str(t)] = np.array(ids)
        z['Fragment_stats/Position/'+str(t)] = np.array(positions)
        z['Fragment_stats/Volume/'+str(t)] = np.array(volumes)
        # positions
        # Get the merges from the second, larger threshold
        _, merges, _ = next(gen)

        def merged_position(fragments, a, b, t):
            """
            Merge 'a' and 'b' into 'w'

            return a center point: tuple float
            """
            merge_mask = np.zeros((fragments.shape), dtype='int')
            merge_mask[seg == a] = 1
            merge_mask[seg == b] = 1
            region = regionprops(merge_mask)
            z, y, x = region[0].centroid
            position = (t, int(z), int(y), int(x))
            # TODO shifting bits can be set by parameter.
            id = encode64((t, int(z), int(y), int(x), int(region[0].area)),bits=[9,12,12,12,19])
            return id, position, region[0].area
        
        def mask(img, label):
            mask = np.zeros((img.shape), dtype='int')
            mask[img == label] = 1
            return mask

        merge_tree = np.empty((len(merges), 3), dtype=np.uint64)
        merge_scores = np.empty((len(merges),))
        # Separately store the stats for the whole merge tree!
        merge_positions = {}
        merge_volumes = {}
        merge_parents = {}

        for i, merge in enumerate(merges):
            # e.g. {a: 1, b: 2, c: 1, score: 0.01}
            # match the ID index 0,1,2,3,..... order
            a, b, c = merge['a'], merge['b'], merge['c']
            score = merge['score']
            #
            f_u = mask(seg, a)
            u, pos_u, vol_u = merge_stats(f_u, t)
            f_v = mask(seg, b)
            v, pos_v, vol_v = merge_stats(f_v, t)
            # Create the merged node
            w, pos_w, vol_w = merged_position(seg, a, b, t)  # index+1 = label
            # merge a,b in to a
            seg[seg == b] = a
            # Add to merge tree
            merge_tree[i] = u, v, w
            merge_scores[i] = score
            # Add to merge stats
            merge_volumes.update({u: vol_u, v: vol_v, w: vol_w})
            merge_parents.update({u: w, v: w, w: w})
            merge_positions.update({u: pos_u, v: pos_v, w: pos_w})

        # add Merge_tree to zarr
        z['Merge_tree/Merge/'+str(t)] = merge_tree
        z['Merge_tree/Scoring/'+str(t)] = merge_scores

        for cell_id in np.unique(merge_tree):
            position = merge_positions[cell_id]
            volume = merge_volumes[cell_id]
            merge_parent = merge_parents[cell_id]
            # Bigger cells are considered "over merged"
            # mangodb can not encode numpy.int try int()
            if volume < max_volume:
                if deploy:
                    cells.insert_one({
                        'id': int(cell_id),
                        'score': float(score),
                        't': int(position[0]),
                        'z': int(position[1]),
                        'y': int(position[2]),
                        'x': int(position[3]),
                        'movement_vector': (0, 0, 0),
                        'parent': int(merge_parent),
                        'size': int(volume)
                    })

    # Add fragments
    z['Fragments'] = np.array(frag_t)


if __name__ == "__main__":
    """

    Example
    -------
    import h5pyÑ
    name='silja_dataset_1_T110_predictions_fused.h5'
    f = h5py.File(name,'r+')
    key = list(f.keys())[0]
    image = f[key][3]

    Channels:
        0: Membrane channel
        1: Wide Field
        2: ZO1 protein
        3: Membrane boundary prediction
    """
    parser = Parser()
    args = parser.parse_args()

    cells = []
    if args.deploy:
        client, cells = create_cells_client(args.name, args.host, args.port,
                                            args.user, args.password)

    z = zarr.open(args.zarrfile, 'a')
    segmentation_process(z, cells, args.deploy, args.max_volume)

    if args.deploy:
        client.close()