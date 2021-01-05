# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds
from scipy.special import softmax

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def get_vis(config, batch_heatmaps, coords):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    vis = np.zeros((batch_size, config.MODEL.NUM_JOINTS, 1))
    heatmaps_reshaped = batch_heatmaps
    idxs = coords
    for i in range(config.MODEL.NUM_JOINTS):
        index_joints = i
        if config.MODEL.USE_HALF_HEATMAP:
            index_joints = i // 2
            if i in {12, 13}:
                index_joints = (i+1) // 2
        for j in range(batch_size):
            heatmaps_tmp = heatmaps_reshaped[j, index_joints]
            vis[j, i] = heatmaps_tmp[idxs[j, i, 1].astype(int), idxs[j, i, 0].astype(int)]
            vis[j, i] = 2 if vis[j, i] >= 0.5 else 1
    return vis
        

def get_final_preds(config, batch_heatmaps, center, scale, batch_heatmaps_vis=None):
    coords, maxvals = get_max_preds(batch_heatmaps)  # coords.shape=(112, 14, 2)
    if batch_heatmaps_vis is not None:
        is_vis = get_vis(config, batch_heatmaps_vis, coords)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )
    if batch_heatmaps_vis is not None:
        return preds, maxvals, is_vis
    
    return preds, maxvals
