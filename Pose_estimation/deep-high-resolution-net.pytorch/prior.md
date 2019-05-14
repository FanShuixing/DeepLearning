## [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212)

1. 数据处理  
  调用模型的train_loader,
```ruby
batch_iter=iter(train_loader)
input,target,target_weight,meta=next(batch_iter)
```

 batch_size=48  
 返回的size分别为(48,3,256,256),(48,16,64,64),(48,16,1),  
 meta返回的value类型为dict,其keys为'images','filename','imgnum','joints','joints_vis','center','scale','rotaton','score'

2. 构建模型   
  ```ruby
    model=models.pose_hrnet.get_pose_net(cfg,is_train=True)
    outputs=model(input)
```

 模型输入:input,size:(batch_size,3,256,256)   
 输出为:outputs，type为tensor,size:(batch_size,16,64,64)
 
3. 损失函数   

 ```ruby
criterion=JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
if isinstance(outputs, list):
        loss = criterion(outputs[0], target, target_weight)
        for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
else:
        output = outputs
        loss = criterion(output, target, target_weight)
```

## Details
### 1.数据处理
Image_process.ipynb有详细步骤

### 2.构建模型



### 3.损失函数
用的是MSE，模型的输出是outputs:(48,16,64,64).target:(48,16,64,64).
```ruby
input:outputs,target
    criterion=nn.MSELoss()

step1: 将 outputs 和 target变换为(48,16,4096) 
step2: make outputs and target ->tuple,tuple里面有16个size为(48,1,4096)的tensor
step3: for i in range(num_joints):
            make outputs and target -> (48,4096)
            loss+=0.5*criterion(outputs,target) or loss+=0.5*criterion(outputs*target_weight,target*target_weight)
            

```

评价指标：PCK  
![images](https://github.com/FanShuixing/DeepLearning/blob/master/Pose_estimation/deep-high-resolution-net.pytorch/imgs/train.png)
上图中为train的时候的log图，其中Accuracy后面有两个值，第一个值为一个batch_size的acc,第二个值是求多个batch_size的acc平均值。acc是在lib/core/evaluate.py的accuracy()函数中求得的。

```
acc计算
input:output:(48,16,64,64),模型预测值
      target:(48,16,64,64),groud truth值
step1:先计算64*64中最大值对应的坐标，返回(48,16,2),(48,16,2)
step2:计算output(48,16,2)和target(48,16,2)的第二范数，返回(16,48)的array，表示每一个关节点对应batch_size张图片个数的第二范数值
step3:对上一步中的(16,48)按行循环遍历，将(48,)计算距离小于thr=0.5的总个数，并除以关节点出现的总数(batch_size或者小于batch_size)，便得到了
      针对某一个关节点，48个样本中距离小于阈值的比例。计算它们的评价值，就得到16个keypoint的平均值，这就是第一个显示的acc的值.
      
```
![images](https://github.com/FanShuixing/DeepLearning/blob/master/Pose_estimation/deep-high-resolution-net.pytorch/imgs/test.png)
上图中Head,Shoulder,Elbow...为对验证集计算pck的值
```
input:pos_gt_src:(16,2,2958),16表示16个keypoint,每一个关键点对应一个坐标，2958表示验证集一共2958张图片，
                             这个坐标是相对于原图，不是相对于模型的输入256*256或64*64
      pos_pred_src:(16,2,2958)，16*2的坐标是模型预测得出的，也是相对于原图
step1:先将pos_gt_src,pos_pred_src做差，得到(16,2,2958),对第二维取第二范数，得到(16,2958),表示2958张图片，每一张图片16个关节点的gt与pred的第二范数距离。
step2:计算scale。gt_dict['headboxes_src']返回的为(2,2,2958),2*2代表头部框的坐标，形式为(x1,y1),(x2,y2)。用[1,:,:]-[0,:,:]得到(2,2958)，
      对列取第二范数，得到(2958,).表示2958张图片每一张图片头部框坐标的第二范数距离.然后将scale变成(16,2958),相当于将2958复制16行
step3:将step1中的值除以step2中的scale,得到(16,2958)，对这个按照行进行统计计算，计算每一行中有多少个小于threshold的值。
      让这个值除以2958(或者小于2958，因为可能有些图的gt中，关节点不可见，只统计关节点可见的情况)，这样就可以得到(16,)的值。
      表示每一个关节点在2958张样本中小于阈值的比例。
    
```
