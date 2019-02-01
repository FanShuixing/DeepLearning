### Deeplab v3+

#### 知识点梳理：

- Depthwise Separable Convolution(mobilenet系列梳理) 
- 空间金字塔池化
- 
--- 
> 参考：[MobileNet v1 和 MobileNet v2](https://zhuanlan.zhihu.com/p/50045821)

deeplab v3代码借鉴了mobilenet v1的深度可分离卷积和mobilenet v2（所以如果有看不懂的代码，可查看mobilenet源码，官方写的很详细）   
**Mobilenet v1**:   
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
    
 ### mobilenet_v2
   > MobileNetV2 is very similar to the original MobileNet,except that it uses inverted residual blocks with bottlenecking features. It has a drastically lower parameter count than the original MobileNet. MobileNets support any input size greater than 32 x 32, with larger image sizes offering better performance.

### 知识点：
  - inverted residual blocks
  - bottlenecking features
  > 参考 [mobilenet v2](https://zhuanlan.zhihu.com/p/33169767)
  
  ![mobilenet v1 and mobilenet v2 structures](https://github.com/FanShuixing/test/blob/master/1/a.jpg)
从源码分析，mobilenet v2在mobilenet v1的基础上做了如下改动（体现在inverted residual blocks结构里面）：
- 添加了一个expansion(1x1的conv2D,BN,relu6)   
  就是上图中右图中的conv 1x1,relu6(在relu6之前应该有BN)
- 当strides=1的时候，多增加了一个类似于resnet 中的residual block的短连接，并且去掉了relu6
  mobilenet的结构有点类似于VGG这种直筒结构，但是Resnet和Densenet的结构证明，复用前面层的特征效果总是好的，所以在mobilenet v2中引入了residual connection的结构，而relu6之前在xception验证了其加在深度可分离卷积层后会损失信息，作者也在mobilenet v2中用大量篇幅推理论证了去掉relu6的必要性。
