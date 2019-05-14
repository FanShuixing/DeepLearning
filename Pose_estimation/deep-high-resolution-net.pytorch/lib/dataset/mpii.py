# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class MPIIDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, 'annot', self.image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        '''
        此方法是被/lib/core/function.py中的validate()函数调用，用于评估验证集的pck值。
        
        step1:从数据中加载matlab文件,/input0/MPII/annot/gt_valid.mat,从mat中读取出需要的值
        step2:计算预测值和真实值之间的距离，pos_pred_src为预测的坐标值，经过变换后size为(16,2,2958),pos_gt_src为真实值，size为(16,2,2958)
              计算两者的差，然后按行计算第二范数，得到的uv_error的size为(16,2958)
        step3:headboxes_src的size为(2,2,2958),2*2表示图片中头部的框，坐标为(x1,y1),(x2,y2)的形式,先计算(x1,y1)和(x2,y2)之间的距离，得到(2,2958)
              大小的矩阵，再对列做第二范数，得到(2958,)大小的矩阵。表示2958张图片，每一张图片头部框坐标(x1,y1),(x2,y2)的第二范数。headsizes*=SC_BIAS。
              计算scale,scale为将headsizes复制16行的数组。大小为(16,2958).每一行表示的都是2958张图片中头部框对应坐标的第二范数值。
        step4:计算scale_uv_error.之间计算出uv_err,size为(16,2958),含义是2958张图片，每一张图片对应的16个关节点的真实值和预测值之间的距离，scale_uv_error
              将uv_err除以scale.
        step5:将scale_uv_error中小于threshold的值按行计数，再除以jnt_count。得到PCKh为(16,)，表示2958张图片中，每一个关节点距离小于threhold的数除以
              这个关节在2958张图片中出现的总数。
        
        input:
            cfg:配置文件
            preds:(imgnums,16,2),preds是模型预测的关节点的坐标，坐标是相对于原图
            output_dir:
            args:接收
            kwargs:接收
            
        output:
            name_value:返回的PCKh中对应的值
            name_value['Mean']:name_value中['Mean']对应的是16个关节点的pck值取平均，['Mean@0.1']是将阈值为0.1,
                               即将距离的threshold从0.5改调为0.1的16个关节点的pck的平均值
            
        '''
        preds = preds[:, :, 0:2] + 1.0
        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds}

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5
        #step1
        gt_file = os.path.join('/input0/MPII/',
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']  #(16,2958)
        #pos_gt_src的值为valid中图像的joints，即关节点的坐标，这个坐标不是相对于255*255或64*64的坐标，而是相对于原图大小的坐标  
        pos_gt_src = gt_dict['pos_gt_src'] #（16，2，2958）
        headboxes_src = gt_dict['headboxes_src']  #(2,2,2958),为头部框的坐标，形式为(x1,y1),(x2,y2)
        #可以通过pos_gt_src中的值得知pos_pred_src中的值为预测出来的关节点的坐标，同样是相对于原图的坐标
        pos_pred_src = np.transpose(preds, [1, 2, 0])
                    
        #如下返回的是索引
        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        #step2
        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src 
        uv_err = np.linalg.norm(uv_error, axis=1) #(16,2958)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :] #(2,2958),(x2,y2)-(x1,y1)
        headsizes = np.linalg.norm(headsizes, axis=0) #(2958,),求二范数，结合上面这步，想当于求(x1,y1),(x2,y2)之间的距离
        headsizes *= SC_BIAS
        #headsizes为(2958,),len(uv_err)=16，scale为(16,2958),相当于将headsizes复制了16行
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))  #(16,2958)
        scaled_uv_err = np.divide(uv_err, scale)  #(16,2958)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)  #(16,)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)  #(16,2958)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count) #(16,)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)  #len(rng)=51
        pckAll = np.zeros((len(rng), 16))  #(51,16)

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
