### [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
**知识点梳理**：

1. Depthwise Separable Convolution 
2. 空洞卷积 
3. ASPP(Atrous Spatial Pyramid Pooling) 

![deeplab v3 model.png](https://github.com/FanShuixing/DeepLearning/blob/master/Semantic%20Segmentation/Deeplab_v3%2B/img/model.png)
上图是deeplab v3+基于xception的模型结构，整个模型是一种encoder-decoder的结构。这种结构在。图中并联的四个卷积和一个image pooling是ASPP结构，
*deeplab v3+代码用了mobilenet v2和xception结构,而xception也是google对inception v3所提出的改进，主要是用了depthwise separable convolution替代了原来的卷积操作。depthwise separable convolution来源于mobilenet.所以下面从mobilenet模型开始梳理。* 

## 1. mobilenet 系列梳理  

### 1.1 Mobilenet v1:   
  mobilenet v1里面主要引入了**Depthwise Separable Convolution**。它的提出就是为了解决传统卷积参数多、计算量大的现象。    
depthwise separable convolution主要包括两部分：depthwise卷积和pointwise卷积。先看传统卷积过程：  
```ruby
inputs = Input(shape=(32,32,10))
x = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(inputs)
```
对于输入为32x32x10,在进行卷积的时候，filter=64,kernel_size=(3,3),我们会用一个3x3x10的滑动窗口在输入的矩阵上滑动做乘法和加法运算，总共有64个这样的filter，最后得到的是32x32x64的特征图。  
现在我们用深度可分离卷积代替上面的传统卷积过程 
```ruby
inputs = Input(shape=(32,32,10))
x = DepthwiseConv2D(kernel_size=(3,3),padding='same', activation='relu', name = 'm_dc_2')(x)    
x = Conv2D(filters=64, kernel_size=(1,1),padding='same', activation='relu', name = 'm_pc_2')(x)
```
1. depthwise卷积   
DepthwiseConv2D没有filters这个参数，因为我们在用DepthwiseConv2D做卷积的时候每个通道对应于单独的卷积核,然后每个通道只和对应的卷积核做乘法，并不会相加。如上面输入32x32x10,DeepthwiseConc2D对应有10个filters，每个filter只和相应的通道做乘法，输出的就是10个通道的特征图，所以没有了普通卷积的跨通道性质，depthwise做的只是一个简单的乘法，并没有合并若干个特征从而产生新的特征，也并没有升维降维的功能。由此引入了pointwise  
2. pointwise卷积（指的就是代码中的第二个卷积操作）    
 pointwise主要做的是用于特征和并以及升维降维，用1x1的filter可以很好的解决这个问题。
    
### 1.2 mobilenet_v2
   > MobileNetV2 is very similar to the original MobileNet,except that it uses inverted residual blocks with bottlenecking features. It has a drastically lower parameter count than the original MobileNet. MobileNets support any input size greater than 32 x 32, with larger image sizes offering better performance.

**知识点**：
  - inverted residual blocks
  - bottlenecking features
  
  ![mobilenet v1 and mobilenet v2 structures](https://github.com/FanShuixing/test/blob/master/1/a.jpg)
从源码分析，mobilenet v2在mobilenet v1的基础上做了如下改动（体现在inverted residual blocks结构里面）：
- 添加了一个expansion(1x1的conv2D,BN,relu6)   
  就是上图中右图中的conv 1x1,relu6(在relu6之前应该有BN)。在resnet的residual block中，通常会用1x1的fliters用于降维，这样可以减少后面3x3卷积的运算量，后面再用1x1的filters升维，但是在inverted residual blocks中，由于我们使用的是深度可分离卷积，深度可分离卷积可以做到在压缩模型参数近8倍的情况下不会过多损伤信息，所以可以通过1x1的filters增加通道数目而不用担心计算量过大。
- 当strides=1的时候，多增加了一个类似于resnet 中的residual block的短连接，并且去掉了relu6   
  mobilenet的结构有点类似于VGG这种直筒结构，但是Resnet和Densenet的结构证明，复用前面层的特征效果总是好的，所以在mobilenet v2中引入了residual connection的结构，而relu6之前在xception验证了其加在深度可分离卷积层后会损失信息，作者也在mobilenet v2中用大量篇幅推理论证了去掉relu6的必要性。

### 1.3 deeplab v3与mobilenet v2:
从源码中可以看出deeplab v3复用了mobinet v2的结构，但是对mobilenet v2中的inverted residual blocks有所改变（对应于源码中的_inverted_res_block函数）
 1. 增加了skip_connection参数，用来指定是否增加residual connection结构。在mobilenet v2中，是通过stides是否等于1来增加residual connection结构。
 2. 增加了rate参数,rate参数是用来指定dilation_rate，这个dilation_rate即是用来指定空洞卷积的膨胀率。
 3. Atrous Spatial Pyramid Pooling   
 使用mobilenet v2作为backbone时，ASPP只有两个分支，使用xception时，ASPP有五个分支，源码中写道尚不清楚为什么要这样做。
 
 ## 2. xception
 ### 2.1 xception结构
 ![xception model.png](https://github.com/FanShuixing/DeepLearning/blob/master/Semantic%20Segmentation/Deeplab_v3%2B/img/xception.png)
 
 ### 2.2 deeplab v3+中的xception结构
 ![deeplab_xveption.png](https://github.com/FanShuixing/DeepLearning/blob/master/Semantic%20Segmentation/Deeplab_v3%2B/img/modified_xception.png)
 ## 2. 空洞卷积  

 ### 2.1 空洞卷积
  ![空洞卷积gif](https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/dilation.gif)
 ### 2.2 空洞卷积作用
 - 扩大感受野
 - 捕获多尺度上下文信息
 
 **扩大感受野**  
 在deep net中为了增加感受野且降低计算量，总要进行降采样(pooling或s2/conv)，这样虽然可以增加感受野，但空间分辨率降低了。为了能不丢失分辨率，且仍然扩大感受野，可以使用空洞卷积。这在检测，分割任务中十分有用。一方面感受野大了可以检测分割大目标，另一方面分辨率高了可以精确定位目标。
 
 **捕获多尺度上下文信息**   
 空洞卷积有一个参数可以设置dilation rate，具体含义就是在卷积核中填充dilation rate-1个0，因此，当设置不同dilation rate时，感受野就会不一样，也即获取了多尺度信息。多尺度信息在视觉任务中相当重要.   
 ### 2.3 空洞卷积存在的问题  
 - griding效应  
 
 ### 3. ASPP 
 ![ASPP structure.png](https://github.com/FanShuixing/DeepLearning/blob/master/Semantic%20Segmentation/Deeplab_v3%2B/img/ASPP.png)
 图中的ASPP截图来源于基于xception结构的deeplab v3+。
**参考**:
> [MobileNet v1 和 MobileNet v2](https://zhuanlan.zhihu.com/p/50045821)  
> [深度学习——分类之MobileNet v2移动端神经网络新选择](https://zhuanlan.zhihu.com/p/33169767)  
> [如何理解空洞卷积（dilated convolution）？](https://www.zhihu.com/question/54149221)    
> [总结-空洞卷积(Dilated/Atrous Convolution)](https://zhuanlan.zhihu.com/p/50369448)  
> [Understanding Convolution for Semantic Segmentation](https://arxiv.org/abs/1702.08502)

