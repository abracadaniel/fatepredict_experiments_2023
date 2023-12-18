# Load FatePredict data into TrackEval compatible format
from pycocotools import mask as mask_utils
import numpy as np

class FatePredictDataset():
    def __init__(self, groundtruth, predicted):
        """
            groundtruth: np.array(T, Z, Y, X) of GT labels
            predicted: np.array(T, Z, Y, X) of predicted labels
        """
        

        # Load raw data.
        raw_gt_data = self._load_raw_data(groundtruth, is_gt=True)
        raw_tracker_data = self._load_raw_data(predicted, is_gt=False)
        raw_data = {**raw_tracker_data, **raw_gt_data}  # Merges dictionaries

        # Calculate similarities for each timestep.
        similarity_scores = []
        for t, (gt_dets_t, tracker_dets_t) in enumerate(zip(raw_data['gt_dets'], raw_data['tracker_dets'])):
            ious = mask_utils.iou(gt_dets_t, tracker_dets_t, [False]*len(tracker_dets_t))
            similarity_scores.append(ious)
        raw_data['similarity_scores'] = similarity_scores

        self.raw_data = raw_data
        self.data = self.get_preprocessed_seq_data(self.raw_data)

    def _load_raw_data(self, data, is_gt):
        # Only loaded when run to reduce minimum requirements
        from pycocotools import mask as mask_utils

        num_timesteps, *_ = data.shape

        data_keys = ['ids', 'dets', 'masks_void']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # read frames
        id_list = []
        for t in range(num_timesteps):
            #print(t)
            frame = data[t]

            id_values = np.unique(frame)
            id_values = id_values[id_values != 0]
            id_list += list(id_values)
            
            tmp = np.ones((len(id_values), *frame.shape))
            if len(frame.shape)==3: # 3D tracking
                tmp = tmp * id_values[:, None, None, None]
            else: # 2D tracking
                tmp = tmp * id_values[:, None, None]

            masks = np.array(tmp == frame[None, ...]).astype(np.uint8)
            if len(frame.shape)==3: # 3D tracking
                _c, _z, _y, _x = masks.shape
                masks = masks.reshape((_c, _z, _y*_x))
            
            _m = np.array(np.transpose(masks, (1, 2, 0)), order='F')

            raw_data['dets'][t] = mask_utils.encode(_m)
            raw_data['ids'][t] = id_values.astype(int)

        num_objects = len(np.unique(id_list))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'dets': 'tracker_dets'}
            
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        raw_data["num_timesteps"] = num_timesteps
        raw_data['mask_shape'] = data[0].shape

        if is_gt:
            raw_data['num_gt_ids'] = num_objects
        else:
            raw_data['num_tracker_ids'] = num_objects

        return raw_data
    
    def get_preprocessed_seq_data(self, raw_data):
        """
        TODO
        """

        # Only loaded when run to reduce minimum requirements
        from pycocotools import mask as mask_utils

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        num_gt_dets = 0
        num_tracker_dets = 0
        unique_gt_ids = []
        unique_tracker_ids = []
        num_timesteps = raw_data['num_timesteps']

        # count detections
        for t in range(num_timesteps):
            num_gt_dets += len(raw_data['gt_dets'][t])
            num_tracker_dets += len(raw_data['tracker_dets'][t])
            unique_gt_ids += list(np.unique(raw_data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(raw_data['tracker_ids'][t]))

        data['gt_ids'] = raw_data['gt_ids']
        data['gt_dets'] = raw_data['gt_dets']
        data['similarity_scores'] = raw_data['similarity_scores']
        data['tracker_ids'] = raw_data['tracker_ids']
        data['tracker_dets'] = raw_data['tracker_dets']

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int64)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int64)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = raw_data['num_tracker_ids']
        data['num_gt_ids'] = raw_data['num_gt_ids']
        data['mask_shape'] = raw_data['mask_shape']
        data['num_timesteps'] = num_timesteps
        return data