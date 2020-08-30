import pandas as pd
import paddlex as pdx
import os
os.chdir('/home/aistudio/work')
# test_jpg = 'mask_r50_fpn_coco/test.jpg'
model = pdx.load_model('output/faster_rcnn_r50_fpn/epoch_12')
# image_name = './image_png/318818_000.png'

#对训练数据fcrnn预测
aa = pd.read_csv('train_list_voc.txt',names=[0,1],sep=' ')
trainFileNameList = list('./' + aa[0])
trainPredictResultList = model.batch_predict(trainFileNameList, transforms=eval_transforms, thread_num=4)

trainPredictResultDF = pd.DataFrame(trainFileNameList)
trainPredictResultDF.to_csv('./train_predict_result.csv')

#生成测试用png 图片
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/aistudio/external-libraries')
import SimpleITK as sitk
from work import generate_the_image
os.chdir('/home/aistudio/work')
file_paths = ['/home/aistudio/data/data8689/testA']
save_path = './test_image_png'
for file_path in file_paths:
    files = generate_the_image.get_file_name(file_path)
    images = generate_the_image.get_all_image(file_path, files, save_path, save_image=True)

#对真实测试数据fcrnn预测
testFileNameList = os.listdir('/home/aistudio/data/data8689/testA')
testFileNameList = list(map(lambda x:'../data/data8689/testA/'+str(x),testFileNameList))
#生成png图片

