
model: deeplab v3+    
data:遥感数据二分类，图片格式tif，images通道：4，masks通道：1  

--  

数据处理： (将图片分割成(224,224,None)的数组，并将所有图片保存在以npy结尾的文件中，在训练的时候直接读取npy)  
- 图片保存路径：./images存放原图，./labels存放mask。  
- mate_data_to_npy.ipynb 将图片转换成npy  
