import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm

import sys
sys.path.append('/home/aistudio/external-libraries')
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg') 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def normalize(image):
    MIN_BOUND = np.min(image)
    MAX_BOUND = np.max(image)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def get_file_id(file_path):
    files = []
    for f_name in [f for f in os.listdir(file_path) if f.endswith('.mhd')]:
        f_name = f_name.replace('.mhd', '')
        files.append(f_name)
    return sorted(files)

def load_itk(file):
    itkimage = sitk.ReadImage(file)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan, origin, spacing



def get_image_and_label(sets=['train_part1'],
                        data_path='../data',
                        anns_path='../data/chestCT_round1_annotation.csv',
                        predict_path=['./cut_csv/lt_yolov3_014_cut_part1.csv'],
                        target_size=64,
                        image_class_save_path = './image_class',
                        image_type_path = '/train',
                        output_file_path = 'train_list.txt'
                        ):
    images, labels = [], []
    anns_all = pd.read_csv(anns_path)
    # predict_anns_all = pd.read_csv(predict_path[i])

    os.chdir('/home/aistudio/work')
    #创建分类图片存储目录
    imageClassSavePath = image_class_save_path + image_type_path
    if not os.path.isdir(imageClassSavePath):
        os.makedirs(imageClassSavePath.replace('./',''))

    #创建分类目录
    imgClassName = ['no','yes']
    for name in imgClassName:
        imageClassDetailPath = imageClassSavePath + '/'+name
        if not os.path.isdir(imageClassDetailPath):
            # print(imageClassDetailPath)
            os.makedirs(imageClassDetailPath.replace('./',''))

    for i, current_set in enumerate(sets):
        predict_anns_all = pd.read_csv(predict_path[i])
        from_path = os.path.join(data_path, current_set)
        file_ids = get_file_id(from_path)
        for current_id in tqdm(file_ids):
            current_file = os.path.join(from_path, current_id + '.mhd')
            ct, origin, spacing = load_itk(current_file)

            #将ct片的世界坐标变成图片中的相对坐标
            ann_df = anns_all.query('seriesuid == "%s"' % current_id).copy()
            ann_df.coordX = (ann_df.coordX - origin[2]) / spacing[2]
            ann_df.coordY = (ann_df.coordY - origin[1]) / spacing[1]
            ann_df.coordZ = (ann_df.coordZ - origin[0]) / spacing[0]
            ann_df.diameterX = ann_df.diameterX / spacing[2]
            ann_df.diameterY = ann_df.diameterY / spacing[1]
            ann_df.diameterZ = ann_df.diameterZ / spacing[0]
            predict_ann_df = predict_anns_all.query('seriesuid == "%s"' % current_id).copy()
            # predict_ann_df = predict_anns_all[predict_anns_all['pic_name'] == './image_png/{}.png'.format(current_id)].copy()
            kk = 0
            for _, predict_ann in predict_ann_df.iterrows():
            #循环预测的结果 找到符合真实标记的数据
                kk = kk + 1
            # for predict_ann in eval(predict_ann_df['predict_result'].iloc[0]):
                pre_minX, pre_minY, pre_z, pre_w, pre_h = predict_ann.minX, predict_ann.minY, predict_ann.coordZ, predict_ann.width, predict_ann.height
                # pre_x, pre_y, pre_z, pre_w, pre_h
                #预测出来的pre_w  和 pre_h 会大一个像素  减少一下就行
                # pre_w = pre_w - 1
                # pre_h = pre_h - 1
                #计算中心节点坐标
                pre_x = pre_minX + pre_w * 0.5
                pre_y = pre_minY + pre_h * 0.5
                flag = False
                for _, ann in ann_df.iterrows():
                #循环所有人为标记的数据
                    x, y, z, w, h, d = int(ann.coordX), int(ann.coordY), int(ann.coordZ), int(ann.diameterX), int(ann.diameterY), int(ann.diameterZ)
                    if pre_z > z - d / 2 and pre_z < z + d / 2:
                        x_min = (x - w / 2)
                        y_min = (y - h / 2)
                        x_max = (x + w / 2)
                        y_max = (y + h / 2)
                        #如果预测的数据中心点  即 0.5*（xmax+xmin） 在手动标记的中间，则认为其标记正确 分类为True
                        if pre_x > x_min and pre_x < x_max and pre_y > y_min and pre_y < y_max:
                            #如果有但凡一个标记正确的 则进行下一个预测标记 的循环
                            # print(int(h),int(w))
                            # print(int(y_min),int(y_max), int(x_min),int(x_max))
                            flag = True
                            current_image = ct[int(pre_z)]
                            max_size = int(max(w, h))
                            result_image = np.zeros((max_size, max_size))

                            result_image[0:int(h), 0:int(w)] = current_image[int(y_min):int(y_max), int(x_min):int(x_max)]
                            result_image = cv2.resize(result_image, (target_size, target_size))
                            # 训练时候会进行normalize  现在不用处理
                            # result_image = normalize(result_image)
                            cv2.imwrite(imageClassSavePath +'/yes/' + str(current_id) + '_' + str(kk).zfill(3)+ '.png', result_image)

                            # images.append('1/'+ str(current_id) + '_' + str(kk).zfill(3)+ '.png')
                            # labels.append(1)
                            # images.append(result_image)
                            # labels.append(flag)
                            break
                #循环完成 说明没有一个人为标记符合
                if flag is False:
                    current_image = ct[int(pre_z)]
                    w, h = int(pre_w), int(pre_h)
                    # pre_x_min, pre_x_max = int(pre_x - pre_w / 2), int(pre_x + pre_w / 2)
                    # pre_y_min, pre_y_max = int(pre_y - pre_h / 2), int(pre_y + pre_h / 2)
                    pre_x_min = int(pre_minX)
                    pre_x_max = int(pre_minX) + int(pre_w)
                    pre_y_min = int(pre_minY)
                    pre_y_max = int(pre_minY) + int(pre_h)
                    # print(int(h),int(w))
                    # print(int(pre_y_min),int(pre_y_max), int(pre_x_min),int(pre_x_max))
                    max_size = int(max(w, h))
                    result_image = np.zeros((max_size, max_size))
                    result_image[0:int(h), 0:int(w)] = current_image[pre_y_min:pre_y_max, pre_x_min:pre_x_max]
                    result_image = cv2.resize(result_image, (target_size, target_size))
                    result_image = np.asarray(result_image).astype(np.float32)
                    # result_image = normalize(result_image)
                    cv2.imwrite(imageClassSavePath +'/no/' + str(current_id) + '_' + str(kk).zfill(3)+ '.png', result_image)
                    # images.append('0/'+ str(current_id) + '_' + str(kk).zfill(3)+ '.png')
                    # labels.append(0)
                    # images.append(result_image)
                    # labels.append(flag)
                


    # images = np.asarray(images)
    # labels = np.asarray(labels)
    # trainFile = open(imageClassSavePath + '/'+ output_file_path,'w')
    # for i in range(len(images)):
    #     trainFile.write(str(images[i])+ ' ' + str(labels) + '\n')
    # trainFile.close()
    os.chdir(imageClassSavePath)
    wrongFileList= os.listdir('./no')
    wrongFileList = list(map(lambda x:'no/'+x+ ' 0',wrongFileList))
    trainFile = open(output_file_path,'w')
    trainFile.write('\n'.join(wrongFileList))
    trainFile.write('\n')
    rightFileList = os.listdir('./yes')
    rightFileList = list(map(lambda x:'yes/'+x+ ' 1',rightFileList))
    trainFile.write('\n'.join(rightFileList))
    trainFile.close()

    # os.makedirs(imageClassSavePath + '/labels.txt')
    f = open('labels.txt','w')
    f.write('no\n')
    f.write('yes\n')
    f.close()
    # return np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1)), np.reshape(labels, (labels.shape[0], 1))

    return images,labels

def DoGenerateResNetImageAndLabel():
    #生成训练数据
    x_train, y_train= get_image_and_label(sets=['train_part1', 'train_part2', 'train_part3', 'train_part4'],
                                    data_path='/home/aistudio/data/data8689',
                                    anns_path='/home/aistudio/data/data8689/chestCT_round1_annotation.csv',
                                    predict_path=['./predict_result/train_predict_frcnn_cut_part1.csv', './predict_result/train_predict_frcnn_cut_part2.csv',
                                                  './predict_result/train_predict_frcnn_cut_part3.csv', './predict_result/train_predict_frcnn_cut_part4.csv'],
                                    target_size=64,
                                    image_class_save_path = './image_class',
                                    image_type_path = '/train',
                                    output_file_path = 'train_list.txt'
                                    )
    #生成测试数据
    x_test, y_test= get_image_and_label(sets=['train_part5'],
                                    data_path='/home/aistudio/data/data8689',
                                    anns_path='/home/aistudio/data/data8689/chestCT_round1_annotation.csv',
                                    predict_path=['./predict_result/train_predict_frcnn_cut_part5.csv'],
                                    target_size=64,
                                    image_class_save_path = './image_class',
                                    image_type_path = '/val',
                                    output_file_path = 'val_list.txt'
                                    )


if __name__ == '__main__':
    #
    DoGenerateResNetImageAndLabel()

    #生成测试数据

    # images, labels = get_image_and_label(sets=['train_part1'],
    #                                     data_path='../data',
    #                                     anns_path='../data/chestCT_round1_annotation.csv',
    #                                     predict_path=['./lt_yolov3_014_cut_part1.csv'],
    #                                     target_size=64,
    #                                     )
    # print(images.shape)
    # print(labels.shape)
    # for i in range(images.shape[0]):
    #     fig, (ax0) = plt.subplots(1, 1)
    #     current_image = images[i]
    #     current_image = np.reshape(current_image, (current_image.shape[0], current_image.shape[1]))
    #     ax0.imshow(current_image)
    #     print(labels[i])
    #     plt.show()
        