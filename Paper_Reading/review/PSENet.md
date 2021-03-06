## PSENet:Shape Robust Text Detection with Progressive Scale Expansion Network

### 1.摘要：
- 挑战：  
  大多数现存的基于检测器的多边形bounding box很难定位任意形状的文本，因为这些任意形状的文本很难被一个矩形所框住。  
  大多数基于分割的逐像素检测器可能并不能将离的很近的文本实例单独区分开来。   

- 本文提出的方法：    
  PSENet,designed as a segmentation-based detector with multiple predictions for each text instance.PSENet是实例分割网络。
  
### 2.Introduction  
**2.1 背景**  
  基于边框回归的方法，提出了许多深度学习模型，它们能够以确定的方向将文本target以矩形或者四边形的形式成功定位，但是这些方法并不能检测任意形状的文本实例（如弯曲的文本）；pixel-wise segmentation虽然能够提取出任意形状文本实例的区域，但是当它们之间的距离很近时，pixes-wise segmentation也很难分离两个文本实例，因为它们相邻的边界是共享的，这可能会导致模型将它们merge为1个实例。
  > Although pixel-wise segmentation can extract the regions of arbitrary-shaped text instances, it may still fail to separate two text instances when they are relatively close, because their shared adjacent boundaries will probably merge them together as one single text instance (see Fig. 1 (c)).
![image](https://github.com/FanShuixing/DeepLearning/blob/master/Paper_Reading/imgs/psenet_1.png)

**2.2 Advantages**:    
1.作为一个基于分割的方法，PSENet能够定位任意形状的文本。   
2.我们提出了一个progressive scale expansion algorithm，这个算法能够成功地鉴定出相邻的文本实例。如Figure 1中图d所示。  

**2.3 Motivation**:  


### 3.Method  
**3.1 Overall Pipeline**:  

![image](https://github.com/FanShuixing/DeepLearning/blob/master/Paper_Reading/imgs/psenet_2.png)
> 上图为PSENet的整体Pipeline,其设计灵感来源于FPN,将low-level的feature maps和high-level的feature map相连接(concatenate)。  
> 这些feature map接下来会融合形成F，F编码了各种尺度的感受野的信息。    
> Then the feature map F is projected into n branches to produce multiple segmentation results S1, S2, ..., Sn. Each Si would be one segmentation mask for all the text instances at a certain scale.
> 在这些mask中，S1表示最小scale的文本实例的分割结果。Sn表示原始的分割mask。  
> 在获得这些分割的mask后，我们使用渐进式尺度扩展算法(progressive scale expansion algorithm)逐渐扩展所有的kernels，从S1到Sn,然后获得最终的检测结果R。

**3.2 Progressive Scale Expansion Algorithm**:

![image](https://github.com/FanShuixing/DeepLearning/blob/master/Paper_Reading/imgs/psenet_3.png)

### 实验：  
