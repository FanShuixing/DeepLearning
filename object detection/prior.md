## [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)


1. 数据处理   
 images,targets=next(batch_iterator),返回的images类型为Tensor,size为(batch_size,3,300,300),targets的类型为list，每个value的类型为Tensor，size=(num_obj,5),5=4+1，4为框的坐标(x_min,y_min,x_max,y_max)，1为框对应的label。
2. 构建模型   
 模型搭建主要是在ssd.py中    
  ```ruby
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net=ssd_net
    out=net(images)
```

 模型输入:images,size:(batch_size,3,300,300)   
 输出为:out，tuple   
 > tuple0.size:(batch_size,8732,4)   
 tuple1.size:(batch_size,8732,num_classes)  
 tuple2.size:(8732,4)  
 
3. 损失函数   

 ```ruby
criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
False, args.cuda)
 loss_l, loss_c = criterion(out, targets)
```

## Details
### 2.构建模型

### 3.损失函数
损失函数分为两部分计算，分类和定位.损失函数的定义主要是在/layers/modules/multibox_loss.py中。假定batch_size=15
```ruby
step1:
for i in range(15):
        #将一张图片的框和lable传入到match中,defaults为default box(prior boxes,8732的先验框)
        match(self.threshold, truths, defaults, self.variance, labels,loc_t, conf_t, idx)   
进入 match:
  step1.1: #假设输入的第一张图片有3个gt box,计算gt box与default box的交并比，返回overlaps,(3,8732)size大小的Tensor,[0,0]位置表示第0个gt box与第0个default box的交并比
    overlaps = jaccard(truths,point_form(priors))
    
  step1.2: #将truths变成(8732,4).
    '''
    truths里面存放的是gt box的坐标，truths的size为(3,4),overlaps里面存放的是交并比，对overlaps的行取最大值
    比如overlaps中的第一列表示default box中的第一个框与gt box三个框的iou，max(0)会返回这一列的对应的最大iou的gt的索引
    best_truth_idx.size()=(8732,1),eg [[2],[0],[0],[1],...]表示default box第一个框与gt box 第3个框的iou最大，default box第二个框
    与gt box第一个框的iou最大'''
    
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    
    #best_truth_idx.size():经过squeeze后为(8732),truths.size():(3,4),matches.size():(8732,4)
    #现在matches返回的就是对于8732个先验框，每一个先验框最匹配的gt box的坐标
    matches = truths[best_truth_idx]  
  step1.3:
    #matches中的框为(x_min,y_min,x_max,y_max),论文中有提到将matches中的框encode为(center_x,center_y,w,h)
    loc = encode(matches, priors, variances)
  step1.4：
    #根据best_truth_idx返回label,+1是因为将0作为背景类
    conf = labels[best_truth_idx] + 1     
    conf[best_truth_overlap < threshold] = 0  # 将iou小于阈值的框都作为背景
step2:
    '''
    经过了match后，更新了loc_c,conf_t。conf_t里面存放的是(15,8732)的tensor,loc_t里面存放的是(15,8732,4)的tensor.
    '''
    step2.1:#上一步返回的conf_t里面是(15,8732),摘选出conf_t>0的框，即非背景的框，对这些框计算定位损失
     loc_p = loc_data[pos_idx].view(-1, 4)
     loc_t = loc_t[pos_idx].view(-1, 4)
     loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
    step2.2:#计算分类的损失，分类的损失不仅计算正例，还要计算负例的损失。正例即conf_t>0的框，也即非背景的框，负例需要摘选。
    #摘选出负例。
    #2.2.1根据公式计算出 Loss
    loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))  
    #2.2.2 让正例的loss为0,对剩下的Loss排序，返回索引，如第12个位置的Loss最大，则这个位置上的索引值为0
    #之前会对每一个batch的正例的数目进行求和，返回一个(15,1)的tensor,eg:[[20],[30],[10],...],负例的数目设为正例的3倍，即返回[[60],[90],[30],...],
    #第一个batch的负例数目为60，则返回loss中索引数目<60的框，即前60个最大的loss.
    #将正例和负例的框摘出来，计算分类的loss.
    
    
```
简而言之，模型的输出为(15,8732,num_classes),(15,8732,4),(8732,4),gt的类型为list，每一个value的值为(num_objects,5),将gt分成两部分，定位的变成(15,8732,4),分类的变为(15,8732)，然后对于定位的loss，选择非背景类的正例计算损失，对于分类的loss，选择3倍于正例的数据作为负例，将正例和负例的数据摘选出来做分类的损失计算。




