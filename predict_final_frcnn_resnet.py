import sys
sys.path.append('/home/aistudio/external-libraries')
from PIL import Image
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from tqdm import tqdm
import math


#对训练数据进行批量预测
import paddlex as pdx
from paddlex.det import transforms as detTransforms
from paddlex.cls import transforms as clsTransforms



label_dict = {}
label_dict['nodule'] = 1
label_dict['stripe'] = 5
label_dict['artery'] = 31
label_dict['lymph'] = 32

os.chdir('/home/aistudio/work')



def get_file_name(file_path):
    files = []
    for f_name in [f for f in os.listdir(file_path) if f.endswith('.mhd')]:
        files.append(f_name)
    return sorted(files)

def load_itk(file):
    itkimage = sitk.ReadImage(file)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


#加载模型
#对训练数据fcrnn预测
model = pdx.load_model('output/faster_rcnn_r50_fpn/epoch_12')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    # print(image.size)
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    # new_image = 0
    return new_image

def pre_predict_image_process(ctImage):
    model_image_size = (512,512)
    boxed_image = letterbox_image(ctImage, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    # image_data = np.expand_dims(image_data, 0)  
    return image_data

def frcnn_predict(fileName,output_path = './predict_result/',clipmin=-1000, clipmax=600):
    # os.chdir('/home/aistudio/work')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    #加载模型
    #对训练数据fcrnn预测
    # model = pdx.load_model('output/faster_rcnn_r50_fpn/epoch_12')

    eval_transforms = detTransforms.Compose([
        detTransforms.Normalize()
    ])

    save_name = 'temp_frcnn_predict_result.txt'
    origin_save_name = 'origin_predict_result.txt'
    origin_predict_result = []
    # file_path = './predict_result'
    seriesuid, minX, minY, coordZ, width, height, diameterZ, label, probability = [], [], [], [], [], [], [], [], []
    f_name = fileName
    result_id = f_name.replace('.mhd', '')
    # current_file = os.path.join(file_path, f_name)
    current_file = f_name
    ct, origin, spacing = load_itk(current_file)
    ct = ct.clip(min=clipmin, max=clipmax)
    for num in tqdm(range(ct.shape[0])):
        # image = pre_predict_image_process(Image.fromarray(ct[num]))
        # image  = np.expand_dims(ct[num], 0)
        cv2.imwrite('./tempPredictImg.png', ct[num])
        image = './tempPredictImg.png'
        detect_result = model.predict(image,transforms=eval_transforms)
        os.remove('./tempPredictImg.png')
        # print(detect_result)
        for one_result in detect_result:
            result_probability = one_result['score']
            result_label = one_result['category']
            seriesuid.append(result_id)
            label.append(result_label)
            probability.append(result_probability)
            minX.append(one_result['bbox'][0])
            minY.append(one_result['bbox'][1])
            coordZ.append(num)
            width.append(one_result['bbox'][2])
            height.append(one_result['bbox'][3])
            diameterZ.append(1)
            #按照原始格式存储图像预测结果
            origin_predict_result.append({'diameterZ':num,'predict_result':one_result})
    dataframe = pd.DataFrame({'seriesuid': seriesuid, 'minX': minX, 'minY': minY, 'coordZ': coordZ,
                                'width': width, 'height': height, 'diameterZ':diameterZ,
                                'label': label, 'probability': probability})
    columns = ['seriesuid', 'minX', 'minY', 'coordZ', 'width', 'height', 'diameterZ', 'label', 'probability']
    dataframe.to_csv(output_path+save_name, index=False, sep=',', columns=columns)

    testFCRNNPredictResultDF = pd.DataFrame(origin_predict_result)
    testFCRNNPredictResultDF.to_csv(output_path+origin_save_name)
    return 0 

#加载resnet 模型
model1 = pdx.load_model('resnet_output/resnet_50/best_model')

def resnet_predict(fileName,temp_rcnn_predict_file,origin_predict_file,
                    output_path = './predict_result/',
                    clipmin=-1000, clipmax=600):
    #对结果进行假阳性预判
    # #加载resnet 模型
    # model1 = pdx.load_model('resnet_output/resnet_50/best_model')
    # model.summary()
    predict_path = temp_rcnn_predict_file
    predict_anns_all = pd.read_csv(output_path+predict_path)
    target_size = 64
    final_save_name = output_path + 'final_predict_frcnn_ResNet.txt'
    origin_final_save_name = output_path + 'final_origin_frcnn_ResNet.txt'

    seriesuid, minX, minY, coordZ, width, height, label, probability = [], [], [], [], [], [], [], []
    trueProb = []
    # origin_resnet_result_for_draw = pd.DataFrame()
    origin_resnet_result_for_draw = []
    origin_resnetDF = pd.read_csv(output_path+origin_predict_file)

    eval_transforms = clsTransforms.Compose([
        clsTransforms.Normalize()
    ])

    file_ids = fileName.replace('.mhd', '')
    # current_file = os.path.join(data_path, current_id + '.mhd')
    current_file = fileName
    current_id = file_ids
    ct, origin, spacing = load_itk(current_file)
    ct = ct.clip(min=clipmin, max=clipmax)
    predict_ann_df = predict_anns_all.query('seriesuid == "%s"' % current_id).copy()

    kk = 0
    for _, predict_ann in tqdm(predict_ann_df.iterrows()):
    # for _,predict_ann in tqdm(predict_ann_df):
        pre_minX, pre_minY, pre_z, pre_w, pre_h = predict_ann.minX, predict_ann.minY, predict_ann.coordZ, predict_ann.width, predict_ann.height
        #计算中心节点坐标
        current_image = ct[int(pre_z)]

        w, h = int(pre_w), int(pre_h)
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
        cv2.imwrite('./tempPredictResNetImg.png', result_image)
        temp_image = './tempPredictResNetImg.png'
        predict_image = model1.predict(temp_image,transforms=eval_transforms)
        # print(predict_image)
        os.remove('./tempPredictResNetImg.png')
        if predict_image[0]['category_id'] ==0:
            #如果frcnn预测结果大于0.7 并且假阳性False概率小于90% 则认为最终结果
            if predict_ann.probability >=0.7 and predict_image[0]['score']<= 0.9:
                seriesuid.append(current_id)
                minX.append(predict_ann.minX)
                minY.append(predict_ann.minY)
                coordZ.append(int(pre_z))
                width.append(predict_ann.width) 
                height.append(predict_ann.height)
                label.append(predict_ann.label)
                trueProb.append(predict_image[0])
                probability.append(predict_ann.probability)
                #将原始的预测格式存储，以便画图
                # origin_predict_result_temp = origin_resnetDF.query('diameterZ == "%s"' % int(pre_z)).copy()
                # origin_predict_result_temp = origin_resnetDF.iloc[kk].copy()
                # print(predict_ann_df)
                # print(predict_ann_df['predict_result'])
                # origin_resnet_result_for_draw.append(origin_predict_result_temp)
                origin_resnet_result_for_draw.append(int(kk))

        else:
            #如果frcnn预测结果大于0.7 并且假阳性True概率大于90% 则认为最终结果
            if predict_ann.probability >=0.7 and predict_image[0]['score'] >= 0.9:
                seriesuid.append(current_id)
                minX.append(predict_ann.minX)
                minY.append(predict_ann.minY)
                coordZ.append(int(pre_z))
                width.append(predict_ann.width) 
                height.append(predict_ann.height)
                label.append(predict_ann.label)
                trueProb.append(predict_image[0])
                probability.append(predict_ann.probability)
                #将原始的预测格式存储，以便画图
                # origin_predict_result_temp = origin_resnetDF.query('diameterZ == "%s"' % int(pre_z)).copy()
                # origin_predict_result_temp = origin_resnetDF.iloc[kk].copy()
                # print(predict_ann_df)
                # print(predict_ann_df['predict_result'])
                # origin_resnet_result_for_draw.append(origin_predict_result_temp)
                origin_resnet_result_for_draw.append(int(kk))
        kk = kk + 1


    dataframe = pd.DataFrame({'seriesuid': seriesuid, 'minX': minX, 'minY': minY, 'coordZ': coordZ,
                        'width': width, 'height': height,'label': label, 'probability': probability})
    columns = ['seriesuid', 'minX', 'minY', 'coordZ', 'width', 'height', 'label', 'probability']
    dataframe.to_csv(final_save_name, index=False, sep=',', columns=columns)
    dataframe['trueType'] = list(map(lambda x:x.get('category'),trueProb))
    dataframe['trueProb'] = list(map(lambda x:float(x.get('score')),trueProb))
    dataframe.to_csv(final_save_name, index=False, sep=',', columns=columns)

    #存储最终的画图结果
    # print(origin_resnet_result_for_draw)
    # origin_resnet_result_for_draw = origin_resnetDF[origin_resnetDF.index in origin_resnet_result_for_draw]
    testFCRNNPredictResultDF = pd.DataFrame(origin_resnetDF.iloc[origin_resnet_result_for_draw])
    testFCRNNPredictResultDF.to_csv(origin_final_save_name)
    #画图
    # for i in range(origin_resnet_result_for_draw.__len__()):
    #     # origin_resnet_result_for_draw
    #     tempPredictPNGName = output_path+'tempPNG'+str(i)+'.png'
    #     cv2.imwrite(tempPredictPNGName, ct[int(origin_resnet_result_for_draw['diameterZ'][0])])
    #     # os.remove('./tempPredictImg.png')
    #     pdx.det.visualize(tempPredictPNGName, origin_resnet_result_for_draw[i], threshold=0.5,save_dir=output_path)
    return 0 


def drawPredictPic(fileName,origin_final_save_name,output_path='./predict_result/',clipmin=-1000, clipmax=600):
    origin_resnet_result_for_draw= pd.read_csv(output_path+origin_final_save_name)
    ct, origin, spacing = load_itk(fileName)
    ct = ct.clip(min=clipmin, max=clipmax)
    #画图
    for name,group in origin_resnet_result_for_draw.groupby('diameterZ'):
        # print(group)
        # break
        predict_result_list = list(group['predict_result'].apply(lambda x: eval(x)))
        # print(predict_result_list)
        tempPredictPNGName = output_path+'tempPNG'+str(name)+'.png'
        cv2.imwrite(tempPredictPNGName, ct[int(name)])
        # os.remove(tempPredictPNGName)
        pdx.det.visualize(tempPredictPNGName, predict_result_list,threshold=0.5,save_dir=output_path)
        os.remove(tempPredictPNGName)
    return 0 

if __name__ == '__main__':
    os.chdir('/home/aistudio/work')
    frcnn_predict('/home/aistudio/data/data8689/testA/340518.mhd')
    save_name = 'temp_frcnn_predict_result.txt'
    origin_save_name = 'origin_predict_result.txt'
    resnet_predict('/home/aistudio/data/data8689/testA/340518.mhd',save_name,origin_save_name)
    drawPredictPic('/home/aistudio/data/data8689/testA/340518.mhd','final_origin_frcnn_ResNet.txt')
