import cv2
from skimage import io
import json
import matplotlib.pyplot as plt
import numpy as np


def visual_box_with_two_points(json_dir, img_dir, n_class=10):
    # 将框画在了图片上
    with open(json_dir) as fr:
        json_f = json.load(fr)
    # 随机生成n_class个rgb颜色图
    colors = np.random.randint(0, 255, (n_class, 3))
    img = io.imread(img_dir)
    label_to_index = {}
    for i in range(json_f['num_box']):
        # find x0,y0,b_width,b_height for visualizing bbox
        label = json_f['bboxes'][i]['label']
        if label not in label_to_index.keys():
            label_to_index[label] = len(label_to_index)
        display_txt = '%s: %.2f' % (label, 1)
        img_width = json_f['image_width']
        img_height = json_f['image_height']
        x0 = json_f['bboxes'][i]['x_min'] * img_width
        y0 = json_f['bboxes'][i]['y_min'] * img_height
        b_width = (json_f['bboxes'][i]['x_max'] - json_f['bboxes'][i]['x_min']) * img_width
        b_height = (json_f['bboxes'][i]['y_max'] - json_f['bboxes'][i]['y_min']) * img_height

        label_index = label_to_index[label]
        color = colors[label_index]
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.putText(img, label, (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + b_width), int(y0 + b_height)), color, 2)
    #     cv2.imwrite('tmp.jpg',img[:,:,::-1])
    return img


def visual_box_with_four_points(json_dir,img_dir):
    label_to_index = {}
    fr=open(json_dir)
    json_f=json.load(fr)

    #读取并显示图片
    img=io.imread(img_dir)
    img_width=json_f['image_width']
    img_height=json_f['image_height']
    for i in range(json_f['num_box']):
        #find x0,y0,b_width,b_height for visualizing bbox
        label=json_f['bboxes'][i]['label']
        if label not in label_to_index.keys():
            label_to_index[label]=colormap[len(label_to_index)]

        x=json_f['bboxes'][i]['x_arr']
        y=json_f['bboxes'][i]['y_arr']
        x=np.array(x)*img_width
        y=np.array(y)*img_height
        a=np.array([[[x[0],y[0]], [x[1],y[1]], [x[2],y[2]], [x[3],y[3]]]], dtype = np.int32)
        color=label_to_index[label]
        cv2.polylines(img, a, 1, color,thickness=3)
    #     cv2.putText(img, label, (int(x[0]),int(y[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    #     cv2.imwrite('%s_box.png'%file_name,img[:,:,::-1])
    return img