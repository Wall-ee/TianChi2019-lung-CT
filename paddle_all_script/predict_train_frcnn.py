import argparse
from PIL import Image
import os
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from tqdm import tqdm


#对训练数据进行批量预测
import pandas as pd
import paddlex as pdx
import os
from tqdm import tqdm
from paddlex.det import transforms



label_dict = {}
label_dict['nodule'] = 1
label_dict['stripe'] = 5
label_dict['artery'] = 31
label_dict['lymph'] = 32




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

def detect_img_lt(clipmin=-1000, clipmax=600):
    file_path = '../data/testA'
    save_name = './train_predict_result.csv'
    seriesuid, coordX, coordY, coordZ, class_label, probability = [], [], [], [], [], []

    files = get_file_name(file_path)
    for f_name in tqdm(files):
        result_id = int(f_name.replace('.mhd', ''))
        current_file = os.path.join(file_path, f_name)
        ct, origin, spacing = load_itk(current_file)
        ct = ct.clip(min=clipmin, max=clipmax)
        for num in range(ct.shape[0]):
            image = Image.fromarray(ct[num])
            detect_result = yolo.detect_image(image)
            for one_result in detect_result:
                result_probability = one_result[1]
                result_label = int(label_dict[one_result[0]])
                result_x = (one_result[2] + one_result[4]) / 2
                result_x = result_x * spacing[2] + origin[2]
                result_y = (one_result[3] + one_result[5]) / 2
                result_y = result_y * spacing[1] + origin[1]
                result_z = num
                result_z = result_z * spacing[0] + origin[0]
                # print(result_id, result_x, result_y, result_z, result_label, result_probability)
                seriesuid.append(result_id)
                coordX.append(result_x)
                coordY.append(result_y)
                coordZ.append(result_z)
                class_label.append(result_label)
                probability.append(result_probability)
    dataframe = pd.DataFrame({'seriesuid': seriesuid, 'coordX': coordX, 'coordY': coordY, 'coordZ': coordZ, 'class': class_label, 'probability': probability})
    columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability']
    dataframe.to_csv(save_name, index=False, sep=',', columns=columns)
    yolo.close_session()


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

def detect_img_lt_like_annotation(clipmin=-1000, clipmax=600):
    file_paths = ['/home/aistudio/data/data8689/train_part1', '/home/aistudio/data/data8689/train_part2', '/home/aistudio/data/data8689/train_part3', '/home/aistudio/data/data8689/train_part4', 
    '/home/aistudio/data/data8689/train_part5', '/home/aistudio/data/data8689/testA']
    save_names = ['./predict_result/train_predict_frcnn_cut_part1.csv', './predict_result/train_predict_frcnn_cut_part2.csv',
                  './predict_result/train_predict_frcnn_cut_part3.csv', './predict_result/train_predict_frcnn_cut_part4.csv',
                  './predict_result/train_predict_frcnn_cut_part5.csv', './predict_result/train_predict_frcnn_cut_testA.csv']


    os.chdir('/home/aistudio/work')
    if not os.path.isdir('./predict_result'):
        os.mkdir('./predict_result')
    #加载模型
    #对训练数据fcrnn预测
    model = pdx.load_model('output/faster_rcnn_r50_fpn/epoch_12')

    eval_transforms = transforms.Compose([
        transforms.Normalize()
    ])

    for i in range(len(file_paths)):
        file_path = file_paths[i]
        save_name = save_names[i]
        seriesuid, minX, minY, coordZ, width, height, diameterZ, label, probability = [], [], [], [], [], [], [], [], []

        files = get_file_name(file_path)
        for f_name in tqdm(files):
            result_id = int(f_name.replace('.mhd', ''))
            current_file = os.path.join(file_path, f_name)
            ct, origin, spacing = load_itk(current_file)
            ct = ct.clip(min=clipmin, max=clipmax)
            for num in range(ct.shape[0]):
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
        dataframe = pd.DataFrame({'seriesuid': seriesuid, 'minX': minX, 'minY': minY, 'coordZ': coordZ,
                                  'width': width, 'height': height, 'diameterZ':diameterZ,
                                  'label': label, 'probability': probability})
        columns = ['seriesuid', 'minX', 'minY', 'coordZ', 'width', 'height', 'diameterZ', 'label', 'probability']
        dataframe.to_csv(save_name, index=False, sep=',', columns=columns)

FLAGS = None

#resnet 50 的kears 预测结果是是这样的
#https://blog.csdn.net/u013093426/article/details/81166751
#
# Input image shape: (1, 64, 64, 3)
# class prediction vector [p(0),p(1),p(2),p(3),p(4),p(5)] =
# [[9.9407300e-02 5.5689206e-03 8.9319646e-01 8.8972512e-05 1.0773229e-03
#   6.6105300e-04]]
#给出每个类别的概率  
# if predict_image[0][1] + predict_ann.probability + 0.5 - predict_image[0][0] > 0:
# 意思是 如果为True 的概率 + 图片识别概率 +0.5 -图片为假的概率大于0  则是真的  我觉得是扯淡




if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image',
        default=True,
        # dest='flag',
        # action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False,default='',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    print(FLAGS)

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        # detect_img_lt(YOLO(**vars(FLAGS)))
        detect_img_lt_like_annotation(YOLO(**vars(FLAGS)))