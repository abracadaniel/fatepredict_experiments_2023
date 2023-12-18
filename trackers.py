from scipy.spatial import distance
from copy import deepcopy
import numpy as np
import waterz
import cv2
import mahotas as mh
from skimage.measure import regionprops
from funlib.math import encode64
from munkres import Munkres
from tqdm import tqdm

def watershed(image):
    inputimage = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # watershed algorithm set local minima as seed
    minima = mh.regmin(inputimage)
    markers, nr_markers = mh.label(minima)
    fragments = mh.cwatershed(inputimage, markers, return_lines=False)
    fragments = fragments.astype('uint64')

    return fragments

def get_fragments(image, threshold=0.5):
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

    # Do WaterZ
    affs = get_affinities(image)
    gen = waterz.agglomerate(affs, [threshold], fragments=fragments,
                         return_merge_history=False,
                         return_region_graph=False)
    fragments  = next(gen)
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


def mask(img, label):
    mask = np.zeros((img.shape), dtype='uint64')
    mask[img == label] = 1
    return mask


def dice_coefficient(y_true, y_pred):
    return 1 - distance.dice(y_true.flatten(), y_pred.flatten())


def jaccard_coefficient(y_true, y_pred):
    return 1 - distance.jaccard(y_true.flatten(), y_pred.flatten())


def precision_recall(gt, pred, threshold=0.95):
    # Compute IoU for each fragment
    pred_correct = np.zeros((len(np.unique(pred))))
    gt_correct = np.zeros((len(np.unique(gt))))
    ji = np.zeros((len(np.unique(pred))))

    for i, p in enumerate(np.unique(pred)):
        p_mask = mask(pred, p)
        
        for j, g in enumerate(np.unique(gt)):
            gt_mask = mask(gt, g)
            jaccard_ij = jaccard_coefficient(gt_mask, p_mask) # IoU
            
            if jaccard_ij>ji[i]: ji[i] = jaccard_ij
            if pred_correct[i]<1: pred_correct[i] = (jaccard_ij>threshold)
            if gt_correct[j]<1: gt_correct[j] = (jaccard_ij>threshold)

    tp = np.sum(pred_correct) # Correct predictions
    fp = np.abs(tp-len(pred_correct)) # The incorrectly predicted cells
    fn = np.abs(tp-len(gt_correct))  # The groundtruth cells which does not have a correct prediction
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    try: f1 = (2*precision*recall)/(precision+recall)
    except: f1 = 0
    
    return (precision, recall, f1, (pred_correct, gt_correct, ji))

def average_precision(gt, pred, t=0.95):
    c = len(np.unique(gt))
    (precision, recall, f1, (pred_correct, gt_correct, ji)) = precision_recall(gt, pred, t)

    ap = 0
    for i, p in enumerate(pred_correct):
        t_k = (ji[i]>t)
        n = i+1
        
        tp = np.sum(pred_correct[:n])
        fp = np.abs(tp-len(pred_correct[:n]))
        
        p_k = tp/(tp+fp)
        ap += p_k*t_k
        
    return ap/c


def encode_id(mask, t):
    region = regionprops(mask)
    z, y, x = region[0].centroid
    position = (t, int(z), int(y), int(x))
    id = encode64((t, int(z), int(y), int(x), int(region[0].area)),bits=[9,12,12,12,19])

    return id, position


def track_segmented(fragments, track_t=0.05, debug=False):
    max_t, *_ = fragments.shape

    _cells_data = {}
    _cell_ids = {}
    _masks = {}
    _fragments = {}
    for t in range(max_t):
        print('t:', t)
        fragments_t = deepcopy(fragments[t])
        
        _fragments[t] = fragments_t
        _masks[t] = []
        _cell_ids[t] = []

        # Iterate over all unique cell fragment ids
        for i in np.unique(fragments_t):
            mask_c = mask(fragments_t, i)      # Get call mask
            _masks[t].append(mask_c)
            
            id_c, pos = encode_id(mask_c, t)   # Get id and regionprops data
            vol = np.sum(mask_c)
            _cell_ids[t].append(id_c)          # Store relation between cell fragment number and cell id
            
            _cells_data[id_c] = {
                'fragment_i': i,
                'vol': vol,
                't': pos[0],
                'z': pos[1],
                'y': pos[2],
                'x': pos[3],
                'parent': None,
                'parent_i': None,
                'origin': id_c
            }
            if debug: print('-i: %s - %s' % (i,id_c))
            
        if t>0:
            _masks_t0 = _masks[t-1]
            _masks_t1 = _masks[t]
            
            cells_t0, cells_t1 = len(_masks_t0), len(_masks_t1)
            _iou = np.zeros((cells_t0, cells_t1))

            # Compute IoU for all cells between t-1 and t
            if debug: print('- Computing IoU')
            for i, m0 in enumerate(tqdm(_masks_t0)):
                if debug: print('%s / %s' % (i, len(_masks_t0)))
                for j, m1 in enumerate(_masks_t1):
                    _iou[i,j] = jaccard_coefficient(m0, m1)

            for i, j in enumerate(np.argmax(_iou, axis=1)):
                id_c0 = _cell_ids[t-1][i]
                id_c1 = _cell_ids[t][j]
                _cell_t0 = _cells_data[id_c0]
                _cell_t1 = _cells_data[id_c1]
                
                if not _cell_t1['origin']:
                    _cell_t1['origin'] = id_c1      # Start new origin

                # Debugging
                if debug: print(i,j, _iou[i,j], id_c0, id_c1)
                
                if _iou[i,j] < track_t:         # IoU is below threshold for matching
                    if debug: print('--iou too small')
                    continue

                if type(_cell_t1['parent_i']) == int:
                    _i = _cell_t1['parent_i']
                    if debug: print('- already matched: %s (%s) - %s (%s)'  % (_iou[i,j], i, _iou[_i,j], _i))

                    if _iou[i, j] < _iou[_i, j]:
                        if debug: print('-- not replacing')
                        continue
                
                origin = _cell_t0.get('origin')
                _cell_t1['origin'] = origin
                _cell_t1['parent'] = id_c0
                _cell_t1['parent_i'] = i

    # Give each id its own color
    for i, id in enumerate(_cells_data.keys()):
        _cells_data[id]['color'] = i
            
    return _cells_data, _fragments


def track_segmented_hung_cent(fragments, debug=False):
    max_t, *_ = fragments.shape

    _cells_data = {}
    _cell_ids = {}
    _masks = {}
    _fragments = {}
    for t in range(max_t):
        print('t: %s / %s' % (t, max_t))
        fragments_t = deepcopy(fragments[t])
        
        _fragments[t] = fragments_t
        _masks[t] = []
        _cell_ids[t] = []

        # Iterate over all unique cell fragment ids
        for i in np.unique(fragments_t):
            mask_c = mask(fragments_t, i)      # Get call mask
            _masks[t].append(mask_c)
            
            id_c, pos = encode_id(mask_c, t)   # Get id and regionprops data
            vol = np.sum(mask_c)
            _cell_ids[t].append(id_c)          # Store relation between cell fragment number and cell id
            
            _cells_data[id_c] = {
                'fragment_i': i,
                'vol': vol,
                't': pos[0],
                'z': pos[1],
                'y': pos[2],
                'x': pos[3],
                'parent': None,
                'parent_i': None,
                'origin': id_c
            }
            #print('-i: %s - %s' % (i,id_c))
            
        if t>0:
            _masks_t0 = [regionprops(x)[0] for x in _masks[t-1]]
            _masks_t1 = [regionprops(x)[0] for x in _masks[t]]
            
            cells_t0, cells_t1 = len(_masks_t0), len(_masks_t1)
            _iou = np.zeros((cells_t0, cells_t1), dtype=np.float64)

            # Compute IoU for all cells between t-1 and t
            if debug: print('- Computing Dist')
            for i, m0 in enumerate(tqdm(_masks_t0)):
                if debug: print('%s / %s' % (i, len(_masks_t0)))
                c0 = np.array(m0.centroid)

                for j, m1 in enumerate(_masks_t1):
                    c1 = np.array(m1.centroid)
                    _iou[i,j] = np.linalg.norm(c0-c1)

            if debug:
                print('- Running hungarian')
                print(_iou.shape)

            s0, s1 = _iou.shape
            if s0>s1:
                if debug: print('- Transposing')
                best_idxs = Munkres().compute(_iou.transpose())
                best_idxs = [(i, j) for (j, i) in best_idxs]
            else:
                best_idxs = Munkres().compute(_iou)
            
            # This is alot more simple, because the hungarian algorithm only assigns each cell to 1 other
            for (i, j) in best_idxs:
                id_c0 = _cell_ids[t-1][i]
                id_c1 = _cell_ids[t][j]
                _cell_t0 = _cells_data[id_c0]
                _cell_t1 = _cells_data[id_c1]
                
                if not _cell_t1['origin']:
                    _cell_t1['origin'] = id_c1      # Start new origin

                # Debugging
                if debug: print(i,j, _iou[i,j], id_c0, id_c1)
                
                origin = _cell_t0.get('origin')
                _cell_t1['origin'] = origin
                _cell_t1['parent'] = id_c0
                _cell_t1['parent_i'] = i

    # Give each id its own color
    for i, id in enumerate(_cells_data.keys()):
        _cells_data[id]['color'] = i
            
    return _cells_data, _fragments


def track_segmented_hung_earthmover(fragments, debug=False):
    from scipy.stats import wasserstein_distance

    max_t, *_ = fragments.shape

    _cells_data = {}
    _cell_ids = {}
    _masks = {}
    _fragments = {}
    for t in range(max_t):
        print('t: %s / %s' % (t, max_t))
        fragments_t = deepcopy(fragments[t])
        
        _fragments[t] = fragments_t
        _masks[t] = []
        _cell_ids[t] = []

        # Iterate over all unique cell fragment ids
        for i in np.unique(fragments_t):
            mask_c = mask(fragments_t, i)      # Get call mask
            _masks[t].append(mask_c)
            
            id_c, pos = encode_id(mask_c, t)   # Get id and regionprops data
            vol = np.sum(mask_c)
            _cell_ids[t].append(id_c)          # Store relation between cell fragment number and cell id
            
            _cells_data[id_c] = {
                'fragment_i': i,
                'vol': vol,
                't': pos[0],
                'z': pos[1],
                'y': pos[2],
                'x': pos[3],
                'parent': None,
                'parent_i': None,
                'origin': id_c
            }
            #print('-i: %s - %s' % (i,id_c))
            
        if t>0:
            _masks_t0 = _masks[t-1]
            _masks_t1 = _masks[t]
            
            cells_t0, cells_t1 = len(_masks_t0), len(_masks_t1)
            _iou = np.zeros((cells_t0, cells_t1), dtype=np.float64)

            if debug: print('- Computing wasserstein dist')
            for i, m0 in enumerate(tqdm(_masks_t0)):
                if debug: print('%s / %s' % (i, len(_masks_t0)))
                for j, m1 in enumerate(_masks_t1):
                    # Masks for values above zero
                    mask1 = m0 > 0
                    mask2 = m1 > 0

                    # Apply masks
                    filtered_matrix1 = m0[mask1 & mask2]
                    filtered_matrix2 = m1[mask1 & mask2]

                    # Compare and count
                    matching_count = np.sum(filtered_matrix1 == filtered_matrix2)
                   
                    if matching_count>0:
                        _iou[i,j] = wasserstein_distance(m0.flatten(), m1.flatten())
                    else:
                        _iou[i,j] = 2

            if debug:
                print('- Running hungarian')
                print(_iou.shape)

            s0, s1 = _iou.shape
            if s0>s1:
                if debug: print('- Transposing')
                best_idxs = Munkres().compute(_iou.transpose())
                best_idxs = [(i, j) for (j, i) in best_idxs]
            else:
                best_idxs = Munkres().compute(_iou)
            
            # This is alot more simple, because the hungarian algorithm only assigns each cell to 1 other
            for (i, j) in best_idxs:
                id_c0 = _cell_ids[t-1][i]
                id_c1 = _cell_ids[t][j]
                _cell_t0 = _cells_data[id_c0]
                _cell_t1 = _cells_data[id_c1]
                
                if not _cell_t1['origin']:
                    _cell_t1['origin'] = id_c1      # Start new origin

                # Debugging
                if debug: print(i,j, _iou[i,j], id_c0, id_c1)
                
                origin = _cell_t0.get('origin')
                _cell_t1['origin'] = origin
                _cell_t1['parent'] = id_c0
                _cell_t1['parent_i'] = i

    # Give each id its own color
    for i, id in enumerate(_cells_data.keys()):
        _cells_data[id]['color'] = i
            
    return _cells_data, _fragments

def track_segmented_hung(fragments, track_t=0.05, debug=False):
    max_t, *_ = fragments.shape

    _cells_data = {}
    _cell_ids = {}
    _masks = {}
    _fragments = {}
    for t in range(max_t):
        print('t: %s / %s' % (t, max_t))
        fragments_t = deepcopy(fragments[t])
        
        _fragments[t] = fragments_t
        _masks[t] = []
        _cell_ids[t] = []

        # Iterate over all unique cell fragment ids
        for i in np.unique(fragments_t):
            mask_c = mask(fragments_t, i)      # Get call mask
            _masks[t].append(mask_c)
            
            id_c, pos = encode_id(mask_c, t)   # Get id and regionprops data
            vol = np.sum(mask_c)
            _cell_ids[t].append(id_c)          # Store relation between cell fragment number and cell id
            
            _cells_data[id_c] = {
                'fragment_i': i,
                'vol': vol,
                't': pos[0],
                'z': pos[1],
                'y': pos[2],
                'x': pos[3],
                'parent': None,
                'parent_i': None,
                'origin': id_c
            }
            #print('-i: %s - %s' % (i,id_c))
            
        if t>0:
            _masks_t0 = _masks[t-1]
            _masks_t1 = _masks[t]
            
            cells_t0, cells_t1 = len(_masks_t0), len(_masks_t1)
            _iou = np.zeros((cells_t0, cells_t1), dtype=np.float64)

            # Compute IoU for all cells between t-1 and t
            if debug: print('- Computing IoU')
            for i, m0 in enumerate(tqdm(_masks_t0)):
                if debug: print('%s / %s' % (i, len(_masks_t0)))
                for j, m1 in enumerate(_masks_t1):
                    _iou[i,j] = 0-jaccard_coefficient(m0, m1)

            if debug:
                print('- Running hungarian')
                print(_iou.shape)

            s0, s1 = _iou.shape
            if s0>s1:
                if debug: print('- Transposing')
                best_idxs = Munkres().compute(_iou.transpose())
                best_idxs = [(i, j) for (j, i) in best_idxs]
            else:
                best_idxs = Munkres().compute(_iou)
            
            # This is alot more simple, because the hungarian algorithm only assigns each cell to 1 other
            for (i, j) in best_idxs:
                id_c0 = _cell_ids[t-1][i]
                id_c1 = _cell_ids[t][j]
                _cell_t0 = _cells_data[id_c0]
                _cell_t1 = _cells_data[id_c1]
                
                if not _cell_t1['origin']:
                    _cell_t1['origin'] = id_c1      # Start new origin

                # Debugging
                if debug: print(i,j, _iou[i,j], id_c0, id_c1)
                
                origin = _cell_t0.get('origin')
                _cell_t1['origin'] = origin
                _cell_t1['parent'] = id_c0
                _cell_t1['parent_i'] = i

    # Give each id its own color
    for i, id in enumerate(_cells_data.keys()):
        _cells_data[id]['color'] = i
            
    return _cells_data, _fragments


def evaluate_hota(gt, pred):
    from trackeval.metrics.hota import HOTA
    from trackeval_fatepredict import FatePredictDataset

    eval = FatePredictDataset(gt, pred)
    hota = HOTA()
    hota_metrics = hota.eval_sequence(eval.data)

    return hota_metrics, hota

def track2(boundaries, wz_t=0.6, track_t=0.05):
    max_t, *_ = boundaries.shape

    _cells_data = {}
    _cell_ids = {}
    _masks = {}
    _fragments = {}
    for t in range(max_t):
        print('t:', t)
        img = deepcopy(boundaries[t])
        fragments_t = get_fragments(img, wz_t)           # Segment boundaries into fragments
        
        _fragments[t] = fragments_t
        _masks[t] = []
        _cell_ids[t] = []

        # Iterate over all unique cell fragment ids
        for i in np.unique(fragments_t):
            mask_c = mask(fragments_t, i)      # Get call mask
            _masks[t].append(mask_c)
            
            id_c, pos = encode_id(mask_c, t)   # Get id and regionprops data
            if i==0: id_c = 0

            vol = np.sum(mask_c)
            _cell_ids[t].append(id_c)          # Store relation between cell fragment number and cell id
            
            _cells_data[id_c] = {
                'fragment_i': i,
                'vol': vol,
                't': pos[0],
                'z': pos[1],
                'y': pos[2],
                'x': pos[3],
                'parent': None,
                'parent_i': None,
                'origin': id_c
            }
            print('-i: %s - %s' % (i,id_c))
            
        if t>0:
            _masks_t0 = _masks[t-1]
            _masks_t1 = _masks[t]
            
            cells_t0, cells_t1 = len(_masks_t0), len(_masks_t1)
            _iou = np.zeros((cells_t0, cells_t1))

            # Compute IoU for all cells between t-1 and t
            print('-Computing IoU')
            for i, m0 in enumerate(_masks_t0):
                print('%s / %s' % (i, len(_masks_t0)))
                for j, m1 in enumerate(_masks_t1):
                    _iou[i,j] = jaccard_coefficient(m0, m1)

            for i, j in enumerate(np.argmax(_iou, axis=1)):
                id_c0 = _cell_ids[t-1][i]
                id_c1 = _cell_ids[t][j]
                _cell_t0 = _cells_data[id_c0]
                _cell_t1 = _cells_data[id_c1]

                if id_c0 == 0 or id_c1: continue # skip background
                
                if not _cell_t1['origin']:
                    _cell_t1['origin'] = id_c1      # Start new origin

                # Debugging
                print(i,j, _iou[i,j], id_c0, id_c1)
                
                if _iou[i,j] < track_t:         # IoU is below threshold for matching
                    print('--iou too small')
                    continue

                if type(_cell_t1['parent_i']) == int:
                    _i = _cell_t1['parent_i']
                    print('- already matched: %s (%s) - %s (%s)'  % (_iou[i,j], i, _iou[_i,j], _i))

                    if _iou[i, j] < _iou[_i, j]:
                        print('-- not replacing')
                        continue
                
                origin = _cell_t0.get('origin')
                _cell_t1['origin'] = origin
                _cell_t1['parent'] = id_c0
                _cell_t1['parent_i'] = i

    # Give each id its own color
    for i, id in enumerate(_cells_data.keys()):
        _cells_data[id]['color'] = i
            
    return _cells_data, _fragments


def gen_tracks(cells_data, fragments, debug=False):
    _fragments = deepcopy(fragments)
    tracks_data = []
    colors_id = {}

    for i in cells_data.keys():
        c = cells_data[i]
        origin, t, z, y, x, f_i = (c['origin'],c['t'],c['z'],c['y'],c['x'],c['fragment_i'])        
        if debug: print('t: %s, f_i: %s, i: %s, origin: %s' % (t, f_i, i, origin))

        fragments_t = _fragments[t].astype(np.int64)
        if f_i > 0:
            tracks_data.append([origin, t, z, y, x])

            # Change color
            fragments_t[fragments_t == f_i] = origin
        else:
            fragments_t[fragments_t == f_i] = 0

        _fragments[t] = fragments_t

    
    # Change cell ids to fragment color
    __fragments = []

    for t in _fragments.keys():
        if debug: print('t:',t)
        fragments_t = _fragments[t]
        ids = np.unique(fragments_t)
        for a,i in enumerate(ids):
            if i>0:
                color = cells_data[i]['color']
                if debug: print('- id: %s, color: %s:' % (i, color))
                fragments_t[fragments_t == i] = color
                colors_id[color] = i
        
        __fragments.append(fragments_t)

    return tracks_data, np.array(__fragments), colors_id

def extract_labels(labels, raw, boundaries, fragments, tracks_data, colors_id):
    _fragments = []
    _raw = []
    _boundaries = []
    _tracks_data = []
    for i, (r, b, f) in enumerate(zip(raw, boundaries, fragments)):
        _r = np.zeros(r.shape)
        _b = np.zeros(b.shape)
        _f = np.zeros(f.shape)
        
        for l in labels:
            m = mask(f, l)
            _r += m*r
            _b += m*b
            _f += m*l
            track_id = colors_id[l]
            for t in tracks_data:
                if t[0] == track_id:
                    _tracks_data.append(t)

        _raw.append(_r)
        _boundaries.append(_b)
        _fragments.append(_f)
        
        
    
    return np.array(_raw), np.array(_boundaries), np.array(_fragments, dtype='uint'), _tracks_data


#Precision / Recall for tracking between 2 frames
# for each fragment in frame1:
#   get matching fragment ID on GT based on IoU
# for each fragment in frame2:
#   get matching fragment ID on GT based on IoU
# check if the id gotten from frame1 matches the id gotten from frame2

def precision_recall_track(gt, pred, threshold=0.95):
    max_t, *_ = pred.shape

    tp = 0
    fp = 0
    for t in range(max_t):
        print('t:', t)
        if t>0:
            t0 = t-1
            t1 = t

            # match pred & gt labels in t0
            t0_pred = pred[t0]
            t0_gt = gt[t0]
            
            t0_gt_labels = np.unique(t0_gt)
            t0_pred_labels = np.unique(t0_pred)
            
            t0_gt_masks = {}
            for l in t0_gt_labels:
                t0_gt_masks[l] = mask(t0_gt, l)

            t0_pred_masks = {}
            for l in t0_pred_labels:
                t0_pred_masks[l] = mask(t0_pred, l)

            count = len(t0_gt_labels)
            t0_label_connection = {}
            t0_ji = np.zeros((len(t0_pred_labels)))
            
            for i, p in enumerate(t0_pred_labels):                
                print('t0: %s / %s' % (i, count))
                
                # Iterate over unmatched gt labels
                for g in t0_gt_labels:
                    jaccard_ij = jaccard_coefficient(t0_gt_masks[g], t0_pred_masks[p]) # IoU
                    
                    if jaccard_ij>t0_ji[i] and jaccard_ij>threshold:
                        t0_ji[i] = jaccard_ij
                        t0_label_connection[p] = g
                        t0_gt_labels = np.delete(t0_gt_labels, np.where(t0_gt_labels == g)) # No need to iterate over this in the future
                        continue

            # Check if there is a connection
            t1_pred = pred[t1]
            t1_gt = gt[t1]
            for p in t0_label_connection.keys():
                g = t0_label_connection[p]
                
                m1 = mask(t1_pred, p)
                m2 = mask(t1_gt, g)
                if jaccard_coefficient(m1, m2)>threshold:
                    tp+=1
                else:
                    fp+=1

    return tp, fp