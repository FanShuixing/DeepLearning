#### [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

pascal voc 和remote_sense使用的模型都是deeplab v3+，区别在于它们所处理的数据不一样，pascal voc处理的数据是普通的图片，图片存放的格式是  
'''
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
'''
