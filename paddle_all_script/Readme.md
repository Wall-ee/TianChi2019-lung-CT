1,解押所有mhd文件
2，将所有解压后的文件创建硬链接到工作目录  ./train/   下面
3，lt_annotation.py 运行该文件，将训练文件中的 annoation 文件整理成 可供原始切割图像用的 csv文件
4，generate_the_image.py 运行该文件,根据 lt_annotation 生成的文件将ct被标记过病变的生成png 图片以供训练
5，yolotovoc.py 运行该文件， 将lt_annotation 生成的转化成voc格式以供训练
6，用 train_frcnn.py文件进行训练  frcnn模型
7，对训练文件，测试文件用 predict_train_frcnn.py   detect_img_lt_like_annotation函数模型进行预测
8，get_image_and_label.py 使用该文件将所有训练文件的预测出来的图像区域，以及标注出来的图像区域进行图片存储及分类。标记病变区域的真假分类
9，用ResNet_train.py 对刚才生成的病变区域图片和文件进行训练  残差网络
10，用predict_final_frcnn_resnet.py 对测试文件 进行1，frcnn预测， 2对标记出的区域进行真假分类判断  3，画图