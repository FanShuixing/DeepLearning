## FCN

### Environment
- python 3.6
- keras 2.2.4

### Prepare

- Environment


- Data folder tree

```
├── train/
│   ├── images/
│       ├── xxx.jpg
│       └── xxx.jpg
│   ├── masks/
│       ├── xxx.png
│       └── xxx.png
├── validation/
│   ├── images/
│       ├── xxx.jpg
│       └── xxx.jpg
│   ├── masks/
│       ├── xxx.png
│       └── xxx.png
├── test/
│   ├── images/
│       ├── xxx.jpg
│       └── xxx.jpg
│   ├── masks/
│       ├── xxx.png
│       └── xxx.png

```
- 下载 [weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)到当前目录下

### Train
直接运行 lab_01_fcn32s.ipynb

### Prediction
运行 lab_01_fcn32s_predict.ipynb
