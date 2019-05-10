## [SSD](https://github.com/amdegroot/ssd.pytorch)

## Envorinment

- pytorch 1.0.0
- python 3.6

## Prepare:

- 下载[weights](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth),并把它放置在ssd/weights/下。
- Datasets Folder Tree:

```
|--images/
|  |-xxx.jpg
|  |-xxx.jpg
|  .
|  .
|  .
|--labels/
|  |-xxx.json
|  |-xxx.json
|  .
|  .
|  .
|--meta.csv
```

## Need to change:
in data/voc0712.py: Line21修改数据集的类别  
in data/config.py:修改voc中的num_class


## Training
```
python train.py
```

## Testing

run demo/demo.ipynb
