#### inference from: https://tianchi.aliyun.com/forum/postDetail?postId=63795
import os
import numpy as np
import SimpleITK as sitk
import skimage.io as io
import cv2

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def load_itk(file):
    # modified from https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
    itkimage = sitk.ReadImage(file)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

def load_marked_pic(trainFileName):
    f = open(trainFileName)
    pngNameList = []
    for line in f.readlines():
        pngNameList.append(line.split(' ')[0])
    f.close()
    return pngNameList

def get_one_image(seriesuid, file_path, see_image, see_path,markedPICList, clipmin=-1000, clipmax=600):
    seriesuid = str(seriesuid)
    file = os.path.join(file_path, seriesuid + '.mhd')
    ct, origin, spacing = load_itk(file)
    ct = ct.clip(min=clipmin, max=clipmax)

    if see_image:
        if not os.path.exists(see_path): 
            os.mkdir(see_path)
    if markedPICList != []:
        for num in range(ct.shape[0]):
            if see_image: 
                #如果图片不在标注的列表里 则不生成png图片
                if './image_png/' + seriesuid + '_' + str(num).zfill(3) + '.png' in markedPICList:
                    cv2.imwrite(os.path.join(see_path, seriesuid + '_' + str(num).zfill(3) + '.png'), ct[num])
    else:
        #如果没有预先设置好的图片列表 则全部mhd生成png图片
        for num in range(ct.shape[0]):
            if see_image: 
                cv2.imwrite(os.path.join(see_path, seriesuid + '_' + str(num).zfill(3) + '.png'), ct[num])
    # return ct
    # ct.close()
    return 0

def get_file_name(file_path):
    files = []
    for f_name in [f for f in os.listdir(file_path) if f.endswith('.mhd')]:
        files.append(f_name)
    return sorted(files)

def get_all_image(file_path, files, save_path, save_image,trainFileName=None):
    result_image = []
    #获取可用的png图片 防止生成过多无用的图片
    if trainFileName is None:
        markedPICList =[]
    else:
        markedPICList = load_marked_pic(trainFileName)
    # print(markedPICList[:40])
    for f_name in tqdm(files):
        seriesuid = f_name.replace('.mhd', '')
        # images = get_one_image(seriesuid, file_path, save_image, save_path)
        # result_image.append(images)
        images = get_one_image(seriesuid, file_path, save_image, save_path,markedPICList)
        result_image.append(images)
    return result_image


if __name__ == "__main__":
    print(os.getcwd())
    file_paths = ['../data/train_part1', '../data/train_part2', '../data/train_part3', '../data/train_part4', '../data/train_part5']
    save_path = './image_png'
    for file_path in file_paths:
        files = get_file_name(file_path)
        images = get_all_image(file_path, files, save_path, save_image=True)