### [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

pascal voc 和remote_sense使用的模型都是deeplab v3+，区别在于它们所处理的数据不一样，pascal voc处理的数据是普通的图片，图片存放的格式如下所示，在训练的时候用的是keras自带的fit_generator
```
|-- train
  |--images
    |--images1.jpg
    |--images2.jpg
    :
  |--masks
    |--masks1.png
    |--masks2.png
    :
|-- val
  |--images
    |--images1.jpg
    |--images2.jpg
    :
  |--masks
    |--masks1.jpg
    |--masks2.png
    :
```
---
remote_sense中处理的是遥感数据集，格式是tif图，遥感数据集特点：图像大，多通道。  
处理遥感图像：  
1.二分类数据集。由于图像较大，通常要做切割，make_data_to_npy.py中将遥感数据集切割成了224x224的小图，步长为100,数据做了随机打乱处理，mask的shape为(none,224,224,2),这是mask的one_hot形式的数据,可以直接输入到模型中做训练，最后并保存为npy的形式。
2.23分类数据集。23分类的数据集若是将图像的one_hot形式保存为npy的形式是非常耗内存的，因为23个通道，其中有非常多的0。所以不采用上述二分类的保存方法，将图像的mask保存为单通道，即(none,224,224),然后在generator读取数据的时候再返回一个batch_size的one_hot形式，这样不占内存。
3.可以去掉一个通道，将tif图保存为jpg,然后就与1.中的方式类似。
