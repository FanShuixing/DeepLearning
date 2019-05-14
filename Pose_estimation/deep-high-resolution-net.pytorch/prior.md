## [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

## Envorinment

- pytorch 1.0.0
- python 3.6

## Prepare:

按照作者中的readme配置

- Datasets Folder Tree:

```
|--images/
|  |-xxx.jpg
|  |-xxx.jpg
|  .
|  .
|  .
|--annot/
|  |-train.json
|  |-train_val.json
|  |-valid.json
|  |-test.json
|  |-gt_valid.mat
```

## Change:
增加了 tools/predict.py:可以用于对指定图像进行预测，并将预测的坐标在图像上显示出来
```
python tools/predict.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
    TEST.MODEL_FILE output/mpii/pose_hrnet/w32_256x256_adam_lr1e-3/final_state.pth
```


