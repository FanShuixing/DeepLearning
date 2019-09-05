## 8月29号

### Mask RCNN 实验  

### 1.loag_image_gt开始将数据改为meta.csv格式

​	首先从self.load_image开始，然后是self.load_mask，然后是compose_image_meta函数

### 	1.1 self.load_image中image_id是为了加载图像

	### 1.2 self.load_mask中image_id是为了获得

mask rcnn 中inference传播的时候images_per_gpu应设置为1，因为对于的batch_size=1



mask_rcnn_learning 第5次实验：

1.不加载预训练模型

balloon.py中Line362修改为pass

2.不只是训练head，整体一起训练

balloon.py中Line 202,将参数layers修改为all
epochs修改为300



待需解决的问题

- [x] class类别对应

- [ ] mAp计算

- [ ] config位置修改

- [ ] bbox得改一下，现在是用utils中的工具根据mask生成的

  mrcnn_data_util.py中Line 150

  