import numpy as np
import cv2,os
from skimage import io

'''
此代码可用于检测分割图像的像素是否正确
'''

def png2idx(data):
    # 将3通道的mask转换为单通道数组
    data = data.astype('uint32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return idx

def color_to_gray(img_name,s_gray_dir):
    '''
    将彩色图转换为灰度图，并提取图像的分割颜色
    '''
    img=cv2.imread(os.path.join(base_dir,img_name))[:,:,::-1]
    label_idx = png2idx(img)
    for idx in np.unique(label_idx):
        if idx not in class_dict['idx_unique'].keys():
            class_dict['idx_unique'][idx] = len(class_dict['idx_unique'])
            flag = 0
            for row in range(label_idx.shape[0]):
                if flag == 1:
                    break
                for col in range(label_idx.shape[1]):
                    if label_idx[row][col] == idx:
                        class_dict['colormap'].append(list(img[row, col, :]))
                        flag = 1
                        break

    # 将三通道的colormap映射为idx
    cm2lbl = np.zeros(256**3)    # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
    for i, cm in enumerate(class_dict['colormap']):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i    # 建立索引
    # 得到单维的mask
    mask_2D = cm2lbl[label_idx]
    cv2.imwrite('%s/%s'%(s_gray_dir,img_name),mask_2D)
    
def labelVisualize(colormap_dict,img):
    #把二维的索引数组根据索引转换为正常的rgb图片
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(len(colormap_dict)):
        img_out[img == i,:] = colormap_dict[i]
    img_out/=255.

    return img_out

def gray_to_color(img_name,colormap_dict,s_dir,gray_dir):
    '''
    将灰度图转换为rgb
    '''
    gray=io.imread(os.path.join(gray_dir,img_name))
    c25_mask=labelVisualize(colormap_dict,gray)
    io.imsave('%s/%s'%(s_dir,img_name),c25_mask)    
    
    
if __name__=='__main__':
    class_dict={}
    class_dict['idx_unique']={}
    class_dict['colormap']=[]
    base_dir='/input0/Pascal_Voc_2012_Segmentation/label/'
    s_gray_dir='gray'
    s_c_dir='color'
    os.makedirs(s_gray_dir,exist_ok=True)
    os.makedirs(s_c_dir,exist_ok=True)
    for img_name in os.listdir(base_dir)[:3]:
        color_to_gray(img_name,s_gray_dir)
        print(class_dict)
        gray_to_color(img_name,class_dict['colormap'],s_c_dir,s_gray_dir)