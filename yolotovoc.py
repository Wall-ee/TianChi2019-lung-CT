import copy

import cv2
import os 
from tqdm import tqdm
import sys
sys.path.append('/home/aistudio/external-libraries')
from lxml.etree import Element, SubElement, tostring, ElementTree
os.chdir('/home/aistudio/work')
 
# 修改为你自己的路径
template_file = './anno.xml'
target_dir = './Annotations/'
image_dir = './image_png/'  # 图片文件夹
train_file = 'train.txt'  # 存储了图片信息的txt文件

target_voc_file = 'train_list_voc.txt'



with open(train_file) as f:
    trainfiles = f.readlines()  # 标注数据 格式(filename label x_min y_min x_max y_max)
 
file_names = []
tree = ElementTree()

targetVocFile = open(target_voc_file,'w')
targetVocFile.truncate()
 
for line in tqdm(trainfiles):
    trainFile = line.split()
    file_name = trainFile[0].split('/')[-1]
    # print(file_name)
 
    # 如果没有重复，则顺利进行。这给的数据集一张图片的多个框没有写在一起。
    if file_name not in file_names:
        file_names.append(file_name)
        tree.parse(template_file)
        root = tree.getroot()
        root.find('filename').text = file_name
        # size
        sz = root.find('size')
        im = cv2.imread(image_dir + file_name)#读取图片信息
        sz.find('height').text = str(im.shape[0])
        sz.find('width').text = str(im.shape[1])
        sz.find('depth').text = str(im.shape[2])
        # object 找到object 循环添加
        obj_ori = root.find('object')
        for kk in range(len(trainFile)):
            if kk >0:
                labelDetail = trainFile[kk].split(',')
                # print(labelDetail)
                lable = labelDetail[4]
                xmin = labelDetail[0]
                ymin = labelDetail[1]
                xmax = labelDetail[2]
                ymax = labelDetail[3]

                # # object 因为我的数据集都只有一个框
                # obj = root.find('object')
                obj = copy.deepcopy(obj_ori)  # 注意这里深拷贝
                
                obj.find('name').text = lable
                bb = obj.find('bndbox')
                bb.find('xmin').text = xmin
                bb.find('ymin').text = ymin
                bb.find('xmax').text = xmax
                bb.find('ymax').text = ymax
                root.append(obj)
        root.remove(obj_ori)
        xml_file = file_name.replace('png', 'xml')
        tree.write(target_dir + xml_file, encoding='utf-8')


        #写入train_list_voc文件
        targetVocFile.write("image_png/{} Annotations/{}\n".format(file_name,xml_file))

# ll = open('labels.txt','w')
# for lab in ['nodule','stripe','artery','lymph']:
#     ll.write('lab\n',encoding='utf-8')
# ll.close()

    # else:
    #     lable = trainFile[4]
    #     xmin = trainFile[0]
    #     ymin = trainFile[1]
    #     xmax = trainFile[2]
    #     ymax = trainFile[3]
 
    #     xml_file = file_name.replace('png', 'xml')
    #     tree.parse(target_dir + xml_file)#如果已经重复
    #     root = tree.getroot()
 
    #     obj_ori = root.find('object')
 
    #     obj = copy.deepcopy(obj_ori)  # 注意这里深拷贝
 
    #     obj.find('name').text = lable
    #     bb = obj.find('bndbox')
    #     bb.find('xmin').text = xmin
    #     bb.find('ymin').text = ymin
    #     bb.find('xmax').text = xmax
    #     bb.find('ymax').text = ymax
    #     root.append(obj)
 
    # xml_file = file_name.replace('png', 'xml')
    # tree.write(target_dir + xml_file, encoding='utf-8')

# """
# 分割训练集和验证集
# 分别存储了图片的路径
# """
# from sklearn.model_selection import train_test_split
 
# img_add = 'G:\\dataset\\train.txt'
# data_set = [x.strip() for x in open(img_add).readlines()]
 
# train_X, test_X = train_test_split(data_set, test_size=0.2, random_state=0)
# print(train_X)
# print(test_X)
 
# train_file = open('G:\\dataset\\WJ-data\\train_file.txt', 'w')
# for x in train_X:
#     x = x.split(' ')[0]
#     print(x)
#     train_file.write('G:\\dataset\\train'+x+'\n')
 
# test_file = open('G:\\dataset\\WJ-data\\valid_file.txt', 'w')
# for x in test_X:
#     x = x.split(' ')[0]
#     test_file.write('G:\\dataset\\train'+x+'\n')

# with open('G:\\dataset\\WJ-data\\obj.names', 'w') as f:
#     for i in range(61):
#         f.write(str(i)+'\n')