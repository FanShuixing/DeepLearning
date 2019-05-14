# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    #preds:(48,16,2)
    #target:(48,16,2)
    #normalize:(48,2)
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))  #(16,48)
    for n in range(preds.shape[0]): #48 
        for c in range(preds.shape[1]): #16
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    
    input:
        output:(48,16,64,64)
        target:(48,16,64,64)
    return:
        acc:一维向量，长度为17，第0个位置上的值为avg_acc，其余位置值的计算方式：
            step1:先通过cals_dists计算pred,target之间的距离，返回dists:(16,48)的array
            step2:对dists中的数据进行行遍历,如dists[0]表示一个batch_size对应的keypoint0的预测值和真实值的距离，将dists[0]传入dist_acc中，
                    计算的是 距离度量小于0.5的个数/总个数(batch_size或者小于batch_size的数)
            
        avg_acc:acc长度为17，除第0个位置外，其它位置的数据表示对应于一个keypoint，batch_size个数据的正确率，avg_acc是对acc中的数求平均所得
        cnt:对acc大于0的数计数
        pred:
    '''
    idx = list(range(output.shape[1])) #一维向量，len(idx)=16
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)  #(48,16,2)
        target, _ = get_max_preds(target)  #(48,16,2)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10  #(48,2)
    dists = calc_dists(pred, target, norm)  #(16,48)

    acc = np.zeros((len(idx) + 1))  #一维0向量，len(acc)=17
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


