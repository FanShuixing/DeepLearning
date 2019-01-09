### [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211)
---  
  
![Image text](https://github.com/FanShuixing/test/blob/master/1.png)  
FCN通过把全连接层转换为卷积层，实现的是一个端到端，像素到像素的密集预测。经典的用于分类的卷积网络，如LeNet,AlexNet,VGG等，它们都需要固定尺寸的输入，通常后面会跟着全连接层，全连接层使得网络的输出没有空间信息。

---
#### 主要技术：  
- 卷积化
- 上采样  
- skip architecture  

  **卷积化**  
将CNN中的全连接层转化为对应的卷积层，如下图所示
 ![image](https://github.com/FanShuixing/test/blob/master/9.png)


  **上采样：**
  - deconvolution 反卷积（转置卷积更恰当)
  - bilinear interpolation(双线性插值)  

  反卷积可以视为将卷积的前向和反向传播过程颠倒，下面是卷积的前向传播：
 
 ![image](https://github.com/FanShuixing/test/blob/master/4.png)

  input=4x4,output=2x2,kernel size=(3,3),stride=(1,1),padding='valid'
  将输入拉长为16x1的向量，输出拉长为4x1的向量，可以将前向传播的过程表示为矩阵乘法 y=C*x,其中C对应是下图中的4x16的稀疏矩阵，在反向传播的过程中，就    是在输出矩阵左乘C的转置矩阵
 
 ![image](https://github.com/FanShuixing/test/blob/master/5.png)
 
  **deconvolution 反卷积（转置卷积更恰当)**  
  反卷积对于输入为2x2的矩阵，输出为4x4的矩阵。将输入拉长为4x1的向量，输出拉长为16x1的向量，可以将转置卷积的前向传播过程视为在输入矩阵左乘C的转置，其反向过程就是左乘C的转置的转置，即乘以C。
 
  **bilinear interpolation(双线性插值)**  
  双线性插值也是一种上采样方法，其原理作者在原文中提到过双线性插值是通过一个线性映射从相邻的四个点中计算出yij  
 
 ![image](https://github.com/FanShuixing/test/blob/master/6.jpg)
 
  上图中是线性插值，即已知两个点，(x0,y0),(x1,y1),y在x点的值，可以通过线性公式得到y的值，双线性插值是分别在x和y方向上做了一次线性插值。
 
 ![image](https://github.com/FanShuixing/test/blob/master/7.png)
 
  如上图，已知图中四个点的值，若想知道未知函数f在点p(x,y)处的值，可以先在x轴方向上做线性插值，在Q12和Q22之间插入点R2，在Q11和Q21之间插入点R1，可以得到f在点R1和点R2的值，再通过点R1和点R2的值，在y轴方向上做线性插值，可以得到f在点p(x,y)处的值，可以发现插值法是不需要学习参数的。

  **skip architecture**

  直接输出最后一层的32倍上采样，结果是很粗糙的，通过添加skips能够融合粗糙的语言信息和局部的表征信息，这种skip architecture通过端到端的学习来优化输出的语义信息,通过skip结构可以更好地预测精细的细节。

![Image](https://github.com/FanShuixing/test/blob/master/3.png)
---
![Image](https://github.com/FanShuixing/test/blob/master/8.png)

  图中主要有三种模型，分别是FCN32s,FCN16s,FCN8s。其中分类选择的是VGG16的结构，VGG16主要是五个卷积块和三个全连接层，分别对应的是fc_4096,fc_4096,fc_1000,pascal voc数据集是21分类(包括类别)，先扔掉fc_1000,然后将fc全连接层卷积化，对应的就是上图中的conv6-7

  - FCN32s:在conv6-7之后接一个卷积层，卷积层的kernel size为(1,1),通道数为类别数目21。VGG每个卷积块通过最大池化层都会降低图片的尺寸为原来的1/2，FCN32s是直接对最后一层进行32倍的上采样，这样可以使输入和输出的尺寸一样，通道数目是类别数目。  
  - FCN16s:在pool4后接一个kernel size为1x1的卷积，通道数目为21的分类层，pool4将输入图片缩小了16倍。conv7的输出是对原图缩小了32倍，在conv7后增加一个kernel size为1x1，通道为21的卷积层用于分类，将conv7后的分类层进行2倍上采样，将pool4与conv7后的卷积层相加，然后进行16倍上采样，就得到了与原图尺寸一样的输出。
  - FCN8s:pool3将输入的尺寸缩小了八倍，在pool3后面添加一个kernel size  为1x1，通道数目为21的卷积层用于分类，pool4和pool5后面也分别添加一个用于分类的1x1卷积层。首先将pool5后的卷积层进行2倍上采样，与pool4后的分类卷积层相加，然后将相加后的卷积层进行2倍上采样，然后再与pool3后的卷积层相加，再把相加后的卷积层进行8倍上采样。
  
不同上采样结果：  

![Image](https://github.com/FanShuixing/test/blob/master/10.png)


#### 实验:
dataset: pascal voc 2012 (segemeantation class 21个类)   
[论文源码](https://github.com/shelhamer/fcn.berkeleyvision.org)   
原作者使用的是caffe,本文是keras,将数据集划分成train，validation,test，比例为8:1:1。每一个文件夹下都存放了images和mask文件夹.    
- 对比了上采样选择转置卷积和双线性插值，转置卷积val_loss下降的速度很慢，以0.00001的速度下降    
- 有无数据增强效果相差不大   
- optimizer选择'adam'的时候val_loss不下降，train的loss和accuracy无规律。选择SGD的时候val_loss下降的很慢，以千分点下降,选择RMSprop效果最好 
- 在处理图片的时候，直接resize图片和对图片进行裁剪效果不一样,经过crop的图片训练出的模型在预测的时候更加精细，小数据集很适合用于裁剪
