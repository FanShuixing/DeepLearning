## FCN

### Environment
- python 3.6
- keras 2.2.4

### Prepare

- Data folder tree

```
├── images/
│       ├── xxx.jpg
│       └── xxx.jpg
├── labels/
│       ├── xxx.png
│       └── xxx.png
├── meta.csv

```
- 下载 [weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)到当前目录下

### Train
直接运行 lab_01_fcn32s.ipynb，注意一定要对图像进行归一化处理，没有归一化的时候，做predict会是一片黑。

### Prediction
运行 lab_01_fcn32s_predict.ipynb
