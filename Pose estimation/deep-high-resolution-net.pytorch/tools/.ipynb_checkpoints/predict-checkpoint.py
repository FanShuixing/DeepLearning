# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from utils.transforms import *

import dataset
import models
import json,cv2
from skimage import io,draw
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    img_root_dir='/input0/MPII/images'

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')


    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=(0,)).cuda()

    img=torch.randn(1,3,256,256)
    output=model(img)
    print(output.size())

    #加载数据
    with open('/input0/MPII/annot/valid.json') as fr:
        valid=json.load(fr)
    print('一共有%d张图片'%len(valid))
    for i in range(10):
        a=valid[i]
        img_name=a['image']
        img_path='%s/%s'%('/input0/MPII/images',img_name)
        c = np.array(a['center'], dtype=np.float)
        s = np.array([a['scale'], a['scale']], dtype=np.float)
        if c[0] != -1:
            c[1] = c[1] + 15 * s[1]
            s = s * 1.25
        c=c-1
        r=0
        #仿射变换
        trans=get_affine_transform(c,s,r,output_size=[256,256])
        print(trans.shape)
        
        data_numpy=io.imread('%s/%s'%(img_root_dir,img_name))
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(256), int(256)),
            flags=cv2.INTER_LINEAR)
        
        input_2 = cv2.warpAffine(
            data_numpy,
            trans,
            (int(256), int(256)),
            flags=cv2.INTER_LINEAR)
        
        print(input.shape)
        
        #仿射变换后，关节点的坐标应发生相应的变换
        joints=np.array(a['joints'])
        joints_vis=np.array(a['joints_vis'])
        num_joints=16
        joints_gt= np.zeros((num_joints,  3), dtype=np.float)
        joints_vis_gt=np.zeros((num_joints,  3), dtype=np.float)
        
        joints_gt[:,0:2]=joints[:,0:2]-1
        joints_vis_gt[:,0]=joints_vis[:]
        joints_vis_gt[:,1]=joints_vis[:]
        
        for i in range(16):
            if joints_vis_gt[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        #先画真实的label图
        for each_point in joints:

            rr,cc=draw.circle(each_point[1],each_point[0],2)

            draw.set_color(input,[rr,cc],[0,255,0])
        io.imsave('%s/%s_%d.jpg'%('save_img',img_name.split('.')[0],i),input)
        
        #预测，需要将numpy转换成tensor
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        input_t=transforms.Compose([transforms.ToTensor(),normalize])(input)
        input_t=torch.unsqueeze(input_t,0)
        print('input_t.size():',input_t.size())
        predict_img=model(input_t)
#         print(predict_img.size())
        
        #将predict_img：(1,16,64,64),转换为(16,2)的坐标，以显示预测出来的图
        #lib/core/inference.py里面的get_max_preds函数
        
        batch_heatmaps=predict_img.detach().cpu().numpy()  #batch_heatmaps的类型应为numpy.ndarry()
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
        preds*=4
        print(preds.shape)
        #画出预测图
        for each_point in preds[0,:,:]:

            rr,cc=draw.circle(each_point[1],each_point[0],2)

            draw.set_color(input_2,[rr,cc],[0,255,0])
        io.imsave('%s/%s_%d_pre.jpg'%('save_img',img_name.split('.')[0],i),input_2)
        
        
        




if __name__ == '__main__':
    main()
