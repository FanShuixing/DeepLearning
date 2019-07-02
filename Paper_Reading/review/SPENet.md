## SPENet:Shape Robust Text Detection with Progressive Scale Expansion Network

### 摘要：
- 挑战：  
  1.大多数现存的基于检测器的多边形bounding box很难定位任意形状的文本，因为这些任意形状的文本很难被一个矩形所框住。  
  2.大多数基于分割的逐像素检测器可能并不能将离的很近的文本实例单独区分开来。   

- 本文提出的方法：    
  PSENet,designed as a segmentation-based detector with multiple predictions for each text instance.PSENet是实例分割网络。
  
### Introduction
  基于边框回归的方法，提出了许多深度学习模型，它们能够以确定的方向将文本target以矩形或者四边形的形式成功定位，但是这些方法并不能检测任意形状的文本实例
  （如弯曲的文本）；pixel-wise segmentation虽然能够提取出任意形状文本实例的区域，但是当它们之间的距离很近时，pixes-wise segmentation也很难分
  离两个文本实例，因为它们相邻的边界是共享的，这可能会导致模型将它们merge为1个实例。
  > Although pixel-wise segmentation can extract the regions of arbitrary-shaped text instances, it may still fail to separate two text instances when they are relatively close, because their shared adjacent boundaries will probably merge them together as one single text instance (see Fig. 1 (c)).
![image]()

**Two Advantages**:  
1.作为一个基于分割的方法，PSENet能够定位任意形状的文本。   
2.我们提出了一个progressive scale expansion algorithm，这个算法能够成功地鉴定出相邻的文本实例。如Figure 1中图d所示。  

**Details**:

**Motivation**:

### 实验：
