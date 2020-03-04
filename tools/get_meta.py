import re
import os
import csv

def main(base_dir,subset_dir,is_train):
    if is_train:
        fr=open('%s/train_meta.csv'%base_dir,'w+')
    else:
        fr = open('%s/val_meta.csv' % base_dir, 'w+')
    writer=csv.writer(fr)
    writer.writerow(['json_Label','image_Source'])
    img_name_list=os.listdir(os.path.join(base_dir,subset_dir,'images'))
    #get suffix
    suffix=img_name_list[0].split('.')[-1]
    for each_name in img_name_list:
        each_name=re.sub(suffix,'',each_name)
        if each_name:
            # print (each_name)
            list=[]
            list.append(os.path.join(subset_dir,'labels',each_name+'json'))
            list.append(os.path.join(subset_dir,'images',each_name+suffix))
            print(list)
            writer.writerow(list)
    fr.close()

if __name__=='__main__':
    base_dir='/Users/pan/Downloads/dataset/pascal_0712_detection_V2'
    # subset_dir='train_0712'
    # main(base_dir,subset_dir,True)
    subset_dir='test_2007'
    main(base_dir,subset_dir,False)