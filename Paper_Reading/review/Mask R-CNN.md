## Mask R-CNN

### 1.摘要

​	Mask RCNN，一种object instance segmentation方法，Mask RCNN能够有效地检测出图像中的objects，并同时为每一个实例生成高质量的分割mask。

### 2.Introduction  

​	Mask RCNN是Faster RCNN的扩展版本，在Faster RCNN的基础上，添加一个分支，为每一个Region of Interest(RoI)预测分割的mask，这个分支是同Faster RCNN中分类和边框回归的分支并行的。
​	ROIAlign的提出：首先，ROIAlign可以保存精确的空间位置信息，ROIAlign可以提高mask accuracy大约10%～50%。其次，我们发现分离mask和分类的预测是必须的。

![imgs](https://github.com/FanShuixing/DeepLearning/blob/master/Paper_Reading/imgs/mrcnn1.png) 

**2.1 Motivation**:  

​	本文旨在为实例分割开发一个可比较的框架。

> Our goal in this work is to develop a compar ably enabling framework for instance segmentation.

### 3.Method  

**3.1 Overall Pipeline**:  
	首先简单介绍下Faster R-CNN，Faster R-CNN是two-stage。第一个stage:RPN，提出候选目标框。第二个stage实际上就是Fast R-CNN，使用RoIPool从每一个候选框提取特征并执行分类和边框回归。 两个阶段的features是可以共享的以达到faster inference的目的。
	Mask R-CNN同样采用two-stage。第一个stage同Faster R-CNN;第二个stage Mask R-CNN添加对每一个RoI输出binary mask的分支。

**3.2 the key elements of Mask R-CNN:pixel-to-pixel alignment**

​		Mask R-CNN引入的mask分支不同于分类和边框回归的输出，需要提取目标的更加精确的空间位置信息。第三个分支mask的loss的定义是使得网络能够对每一个类别产生mask，然后使用sigmoid和binary cross entropy，这样就避免了类别之间的竞争。实验证明以上这一点是能够产生好的实例分割的关键（不同于FCN，FCN通常执行逐像素类别分类，使用multinomial cross entropy,这样会导致mask在类别之间竞争）
​		文章中使用FCN为每一个RoI预测一个mxm的mask，这种pixel-to-pixel需要RoI features能够很好地对齐，以保存每一个像素的空间相关性。为此，作者提出了RoIAlign， 以解决RoIpool引入了misalignment问题。具体可看参考链接   
![RoIAlign](https://github.com/FanShuixing/DeepLearning/blob/master/Paper_Reading/imgs/mrcnn2.png)


### 参考:

1.[详解 ROI Align 的基本原理和实现细节](http://blog.leanote.com/post/afanti.deng@gmail.com/b5f4f526490b)



