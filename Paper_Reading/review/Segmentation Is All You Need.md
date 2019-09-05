## 题目

### 1.摘要

- 挑战：  基于区域搜索的方法在目标检测中效果较好，但是在一些特殊的环境中，这些方法的检测效果就不如人意（比如遮挡、大量的小目标物体）。这主要是因为边框回归包含了许多环境噪声信息，还有一个原因是因为使用了NMS来选择目标框。
- 本文提出的方法：  本文提出了weakly supervised multimodal annotation segmentation（WSMA-Seg），这是第一个anchor-free和NMS-free的目标检测方法。WSMA-Seg是利用分割模型来实现精确的、鲁棒的目标检测模型（并且没有NMS）。WSMA-Seg提出了多模态分割标注来实现实例分割。使用MSP-Seg来作为分割网络，同时提出了一个run-data-based following算法用以追踪目标的轮廓。

### 2.Introduction  

**2.1 背景**  

![image-20190904203033809](https://github.com/FanShuixing/DeepLearning/blob/master/Paper_Reading/imgs/seg1.png)	

文中指出，在上述情况下使用region proposal机制会遇到两个难点：(1)region proposal 机制的性能严重依赖于purity of bounding boxes，然而在上述这些场景下，标注的边框常常比普通场景下的边框包含了非常多的环境噪声  (2) 在region proposal 机制中常常会使用NMS来选择目标框(通过设定IoU来过滤掉其它目标框)，但是很难找到合适的阈值来使用一些复杂的情况。

### 3.Method  

**3.1 Overall Pipeline**:  

![image-20190904210903356](https://github.com/FanShuixing/DeepLearning/blob/master/Paper_Reading/imgs/seg2.png)

WSMA-Seg有两个阶段：训练和测试阶段。

训练阶段-多模态标注数据：WSMA-Seg首先将弱监督边框标注转换为带有3个通道的逐像素分割mask，如上图中所示，三个通道分别表示内部，边界，交集的边界，所以称之为多模态标注。这三个根据bounding box生成的标注将作为WSMA-Seg分割的label，从而训练分割网络。(值得注意的是，作者是根据bounding box产生多模态分割标注图，具体是通过bounding box生成内切椭圆来表示实例，因为作者认为基于像素的分割信息并没有在分割模型中得到充分地利用，所以作者认为精确的分割标注可能对于实现一个比较合理的性能来说并不是很重要）

训练阶段-分割网络：

![image-20190905100403865](https://github.com/FanShuixing/DeepLearning/blob/master/Paper_Reading/imgs/seg4.png)

分割网络是在Hourglass上进行修改，主要是引进了一个mutli-scale block，用来执行multi-scale pooling，如下图所示

![image-20190905101303644](https://github.com/FanShuixing/DeepLearning/blob/master/Paper_Reading/imgs/seg5.png)

测试阶段：

![image-20190904211632268](https://github.com/FanShuixing/DeepLearning/blob/master/Paper_Reading/imgs/seg3.png)

如上图所示，首先输入图片到训练好的分割网络中，产生多模态Heatmap，然后基于一个像素级别的操作，这三个heatmap将转换为一个实例分割图。最后，使用这个分割图来生成目标的轮廓(作者提出使用RDB算法来获取轮廓，简单来说就是需要两个变量来存放轮廓像素的位置，若一个像素左边为0右边为1，则它为左轮廓，反之则为右轮廓)，然后目标的边框就是由轮廓的外切四边形组成。

