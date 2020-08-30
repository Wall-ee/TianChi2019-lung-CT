import sys
sys.path.append('/home/aistudio/external-libraries')
#图像增强
import os
import paddlex as pdx
from paddlex.det import transforms

train_transforms = transforms.Compose([
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.Normalize()
])

os.chdir('/home/aistudio/work')
train_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/work/',
    file_list='./train_list_voc.txt',
    label_list='./labels.txt',
    transforms=train_transforms,
    shuffle=True)

num_classes = len(train_dataset.labels) + 1
model = pdx.det.FasterRCNN(num_classes=num_classes)
model.train(
    num_epochs=12,
    train_dataset=train_dataset,
    train_batch_size=2,
    # eval_dataset=eval_dataset, 
    learning_rate=0.0025,
    lr_decay_epochs=[8, 11],
    save_interval_epochs=1,
    save_dir='output/faster_rcnn_r50_fpn',
    use_vdl=True)