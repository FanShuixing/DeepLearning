### Deeplab v3+

#### 知识点梳理：

-
Mobilenet v1:  
mobilenet v1里面主要引入了**Depthwise Separable Convolution**。 
depthwise separable convolution主要包括两部分：depthwise卷积和pointwise卷积。  
深度可分离卷积的提出就是为了解决传统卷积参数多、计算量大。    
深度可分离卷积将传统卷积分为两个步骤进行。先看传统卷积过程：  
```ruby
    inputs = Input(shape=(32,32,10))
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(inputs)
```
对于输入为32*32*3,在进行卷积的时候，filter=64,kernel_size=(3,3),我们会用一个3*3*10的滑动窗口在输入的矩阵上滑动做乘法和加法运算，总共有64个这样的filter，最后得到的是32*32*64的特征图。  
现在我们用深度可分离卷积代替上面的传统卷积过程
```ruby
    inputs = Input(shape=(32,32,10))
    x = DepthwiseConv2D(kernel_size=(3,3),padding='same', activation='relu', name = 'm_dc_2')(x)    
    x = Conv2D(filters=64, kernel_size=(1,1),padding='same', activation='relu', name = 'm_pc_2')(x)
```
第一步：depthwise卷积
DepthwiseConv2D没有filters这个参数，因为我们在用DepthwiseConv2D做卷积的时候每个通道对应于单独的卷积核,然后每个通道只和对应的卷积核做乘法，并不会相加。如上面输入32*32*10,DeepthwiseConc2D对应有10个filters，每个filter只和相应的通道做乘法，输出的就是10个filters，所以没有了普通卷积的跨通道性质，depthwise做的只是一个简单的乘法，并没有合并若干个特征从而产生新的特征，也并没有升维降维的功能。由此引入了pointwise  
第二步：pointwise    
pointwise主要做的是用于特征和并以及升维降维，用1*1的filter可以很好的解决这个问题。
