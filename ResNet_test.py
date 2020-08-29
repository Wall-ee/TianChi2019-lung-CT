'''
reference from:  https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
'''
import numpy as np
import os
# from get_image_and_label import get_image_and_label
import sys
sys.path.append('/home/aistudio/external-libraries')
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import SimpleITK as sitk
import pandas as pd
# import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
import paddlex as pdx
import json
os.chdir('/home/aistudio/work')

import matplotlib
matplotlib.use('Agg') 
# matplotlib.use('Agg') 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#图像增强
from paddlex.cls import transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.Normalize()
])





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


def DoResnetPredict():
    # model = load_model('./saved_models/trained_weights_final_ResNet.h5')
    model = pdx.load_model('resnet_output/resnet_50/best_model')
    # model.summary()
    predict_path = './predict_result/train_predict_frcnn_cut_testA.csv'
    data_path = '/home/aistudio/data/data8689/testA'
    predict_anns_all = pd.read_csv(predict_path)
    target_size = 64
    save_name = './predict_result/test_file_final_predict_frcnn_testA_ResNet1.csv'
    # seriesuid, coordX, coordY, coordZ, class_label, probability = [], [], [], [], [], []
    seriesuid, minX, minY, coordZ, width, height, label, probability = [], [], [], [], [], [], [], []
    trueProb = []
    file_ids = get_file_id(data_path)
    for current_id in tqdm(file_ids):
        current_file = os.path.join(data_path, current_id + '.mhd')
        ct, origin, spacing = load_itk(current_file)
        predict_ann_df = predict_anns_all.query('seriesuid == "%s"' % current_id).copy()

        # for _, predict_ann in tqdm(predict_ann_df.iterrows()):
        for _, predict_ann in predict_ann_df.iterrows():
            # pre_x, pre_y, pre_z, pre_w, pre_h = predict_ann.coordX, predict_ann.coordY, predict_ann.coordZ, predict_ann.diameterX, predict_ann.diameterY
            pre_minX, pre_minY, pre_z, pre_w, pre_h = predict_ann.minX, predict_ann.minY, predict_ann.coordZ, predict_ann.width, predict_ann.height
            #计算中心节点坐标
            # pre_x = pre_minX + pre_w * 0.5
            # pre_y = pre_minY + pre_h * 0.5
            current_image = ct[int(pre_z)]

            w, h = int(pre_w), int(pre_h)
            # pre_x_min, pre_x_max = int(pre_x - pre_w / 2), int(pre_x + pre_w / 2)
            # pre_y_min, pre_y_max = int(pre_y - pre_h / 2), int(pre_y + pre_h / 2)
            pre_x_min = int(pre_minX)
            pre_x_max = int(pre_minX) + int(pre_w)
            pre_y_min = int(pre_minY)
            pre_y_max = int(pre_minY) + int(pre_h)

            if w > target_size or h > target_size:
                max_size = int(max(w, h))
                result_image = np.zeros((max_size, max_size))
                result_image[0:int(h), 0:int(w)] = current_image[pre_y_min:pre_y_max, pre_x_min:pre_x_max]
                result_image = cv2.resize(result_image, (target_size, target_size))
            else:
                result_image = current_image[pre_y_min:pre_y_max, pre_x_min:pre_x_max]
                result_image = cv2.copyMakeBorder(result_image,
                                                  int((target_size - h) / 2), math.ceil((target_size - h) / 2),
                                                  int((target_size - w) / 2), math.ceil((target_size - w) / 2),
                                                  cv2.BORDER_CONSTANT, value=0)
            # result_image = normalize(result_image)
            cv2.imwrite('./tempPredictResNetImg.png', result_image)
            temp_image = './tempPredictResNetImg.png'
            predict_image = model.predict(temp_image,transforms=eval_transforms)
            os.remove('./tempPredictResNetImg.png')
            # [{'category_id': 0, 'category': 'no', 'score': 0.95524794}]
            # predict_image = model.predict(result_image[np.newaxis, ::, ::],transforms=eval_transforms)

            # predict_image = model.predict(result_image[np.newaxis, ::, ::, np.newaxis])
            # print(predict_image, current_id, pre_x, pre_y, pre_z)
            # fig, (ax0, ax1) = plt.subplots(1, 2)
            # ax0.imshow(current_image)
            # ax1.imshow(result_image)
            # plt.show()
            # if predict_image[0][1] + predict_ann.probability - predict_image[0][0] > 0:
            # if predict_image[0][1] - predict_image[0][0] > 0:
            # if predict_image[0][1] + predict_ann.probability + 0.5 - predict_image[0][0] > 0:
            #     seriesuid.append(int(current_id))
            #     coordX.append(pre_x * spacing[2] + origin[2])
            #     coordY.append(pre_y * spacing[1] + origin[1])
            #     coordZ.append(pre_z * spacing[0] + origin[0])
            #     class_label.append(int(predict_ann.label))
            #     probability.append(predict_ann.probability)
            # if predict_image[0]['category_id'] ==0:
            #     (1- predict_image[0]['score'])+ predict_ann.probability + 0.5 - predict_image[0]['score'] > 0:
            # # if predict_image[0][1] + predict_ann.probability + 0.5 - predict_image[0][0] > 0:
            #         seriesuid.append(int(current_id))
            #         coordX.append(pre_x * spacing[2] + origin[2])
            #         coordY.append(pre_y * spacing[1] + origin[1])
            #         coordZ.append(pre_z * spacing[0] + origin[0])
            #         class_label.append(int(predict_ann.label))
            #         probability.append(predict_ann.probability)
            # else:
            #     (predict_image[0]['score'])+ predict_ann.probability + 0.5 - (1-predict_image[0]['score']) > 0:
            # # if predict_image[0][1] + predict_ann.probability + 0.5 - predict_image[0][0] > 0:
            #         seriesuid.append(int(current_id))
            #         coordX.append(pre_x * spacing[2] + origin[2])
            #         coordY.append(pre_y * spacing[1] + origin[1])
            #         coordZ.append(pre_z * spacing[0] + origin[0])
            #         class_label.append(int(predict_ann.label))
            #         probability.append(predict_ann.probability)
            if predict_image[0]['category_id'] ==0:
                if predict_ann.probability >=0.7 and predict_image[0]['score']<= 0.9:
                    seriesuid.append(int(current_id))
                    # coordX.append(p re_x * spacing[2] + origin[2])
                    # coordY.append(pre_y * spacing[1] + origin[1])
                    # coordZ.append(pre_z * spacing[0] + origin[0])
                    minX.append(predict_ann.minX)
                    minY.append(predict_ann.minY)
                    coordZ.append(int(pre_z))
                    width.append(predict_ann.width) 
                    height.append(predict_ann.height)
                    label.append(predict_ann.label)
                    trueProb.append(predict_image[0])
                    probability.append(predict_ann.probability)
            else:
                if predict_ann.probability >=0.7 and predict_image[0]['score'] >= 0.9:
            # if predict_image[0][1] + predict_ann.probability + 0.5 - predict_image[0][0] > 0:
                    seriesuid.append(int(current_id))
                    # coordX.append(pre_x * spacing[2] + origin[2])
                    # coordY.append(pre_y * spacing[1] + origin[1])
                    # coordZ.append(pre_z * spacing[0] + origin[0])
                    # label.append(predict_ann.label)
                    # probability.append(predict_ann.probability)
                    minX.append(predict_ann.minX)
                    minY.append(predict_ann.minY)
                    coordZ.append(int(pre_z))
                    width.append(predict_ann.width) 
                    height.append(predict_ann.height)
                    label.append(predict_ann.label)
                    trueProb.append(predict_image[0])
                    probability.append(predict_ann.probability)
            #如果真阳性概率大于0.5 并且  预测概率大于0.5  最终确认
            # for aa in predict_image[0]
            # if predict_image[0]['category_id'] == 1:
            #     if 

            # if predict_image[0][1] + predict_ann.probability + 0.5 - predict_image[0][0] > 0:
            # seriesuid.append(int(current_id))
            # coordX.append(pre_x * spacing[2] + origin[2])
            # coordY.append(pre_y * spacing[1] + origin[1])
            # coordZ.append(pre_z * spacing[0] + origin[0])
            # class_label.append(int(predict_ann.label))
            # probability.append(predict_ann.probability)


    # dataframe = pd.DataFrame({'seriesuid': seriesuid, 'coordX': coordX, 'coordY': coordY, 'coordZ': coordZ, 'class': class_label, 'probability': probability})
    # columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability']
    # dataframe.to_csv(save_name, index=False, sep=',', columns=columns)

    dataframe = pd.DataFrame({'seriesuid': seriesuid, 'minX': minX, 'minY': minY, 'coordZ': coordZ,
                            'width': width, 'height': height,'label': label, 'probability': probability})
    columns = ['seriesuid', 'minX', 'minY', 'coordZ', 'width', 'height', 'label', 'probability']
    dataframe.to_csv(save_name, index=False, sep=',', columns=columns)
    dataframe['trueType'] = list(map(lambda x:x.get('category'),trueProb))
    dataframe['trueProb'] = list(map(lambda x:float(x.get('score')),trueProb))
    dataframe.to_csv(save_name, index=False, sep=',', columns=columns)


if __name__ == '__main__':
    DoResnetPredict()