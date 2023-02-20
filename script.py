dir1 = "/usr/volume/WSSS/WSSS_PML/distances_0.txt"
dir2= "/usr/volume/WSSS/WSSS_PML/distances_1.txt"
dists = []
with open(dir1, 'r') as f:
    for line in f.readlines():
        dists.append(float(line.strip('\n')))
with open(dir2, 'r') as f:
    for line in f.readlines():
        dists.append(float(line.strip('\n')))
from matplotlib import pyplot as plt
d = 0.1
num_bins = 10
plt.figure(figsize=(20,8), dpi=80)
plt.hist(dists, bins=10)
xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.xticks(xticks)
plt.grid(alpha=0.4)
plt.savefig("/usr/volume/WSSS/WSSS_PML/distances_hist.jpg")
print(dists)
a =1


'''
    修改图片预处理函数，对图片和gt_Seg同步调整大小
'''
# import torch
# import importlib
# import numpy as np
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from voc12.datacopy import VOC12ClsDataset
# from tool import imutilscopy as imutils
# train_list = "/usr/volume/WSSS/WSSS_PML/voc12/train_aug.txt"
# voc12_root = "/usr/volume/WSSS/VOCdevkit/VOC2012"
# network = "network.resnet38_cls_ser_jointly_revised_seperatable"
# gt_dir = "/usr/volume/WSSS/VOCdevkit/VOC2012/SegmentationClass"

# crop_size = 448
# batch_size = 10
# num_workers = 8

# def worker_init_fn(worker_id):
#         np.random.seed(1 + worker_id)

# model = getattr(importlib.import_module(network), 'Net')()

# train_dataset = VOC12ClsDataset(train_list, voc12_root=voc12_root, gt_root = gt_dir,
#                                             transform=transforms.Compose([
#                                                 imutils.RandomResizeLong(448, 768),   # 随机将长边resize到448-768之间的值，短边自适应
#                                                 transforms.RandomHorizontalFlip(),
#                                                 transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
#                                                                         hue=0.1),
#                                                 np.asarray,
#                                                 model.normalize,
#                                                 imutils.RandomCrop(crop_size),   # 随机裁剪出448*448
#                                                 imutils.HWC_to_CHW,
#                                                 torch.from_numpy
#                                             ]))

# train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
#                                 shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True,
#                                 worker_init_fn=worker_init_fn)

# # for iter, pack in enumerate(train_data_loader):
# #     pass
# #     # pack 再包含分割gt
# #     # 取出之后看大小是否匹配
# import os
# dataset_path = "/usr/volume/WSSS/WSSS_PML/voc12/train_aug.txt"
# list = os.listdir("/usr/volume/WSSS/VOCdevkit/VOC2012/Annotations")
# print(len(list))
# img_gt_name_list = open(dataset_path).read().splitlines()
# img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]
# print("img nums:{}".format(len(img_name_list)))
# cout = 0
# for name in img_name_list:
#     if not f"{name}.xml" in list:
#         print(name)
#         cout+=1
# print(cout)

'''
    找到目标图片保存在当前路径下
'''
# import os
# import numpy as np
# from PIL import Image
# from voc12.data import get_img_path
# name = "2011_002641"
# # 原图
# # Image.open(get_img_path(name, "/usr/volume/WSSS/VOCdevkit/VOC2012")).save(f"./{name}.jpg")
# # 分割结果图
# gt_dir = "/usr/volume/WSSS/VOCdevkit/VOC2012/SegmentationClass"
# gt_file = os.path.join(gt_dir,'%s.png'%name)
# Image.open(gt_file).save(f"./{name}_Seg.png")

'''
    检查val数据集的图像级标签是否正确，与val数据集的gt分割标签对照
'''
# import os
# import numpy as np
# import pandas as pd 
# from PIL import Image
# gt_dir = "/usr/volume/WSSS/VOCdevkit/VOC2012/SegmentationClass"
# label_dir = "voc12/cls_labels.npy"
# img_list = "/usr/volume/WSSS/WSSS_PML/voc12/val.txt"
# cls_labels_dict = np.load('voc12/cls_labels.npy',allow_pickle=True).item()

# df = pd.read_csv(img_list, names=['filename'])
# name_list = df['filename'].values
# for name in name_list:
#     gt_file = os.path.join(gt_dir,'%s.png'%name)
#     gt = np.array(Image.open(gt_file))
#     gt_labels = np.unique(gt)[1:-1]
#     store_labels = np.array(range(1,21))[cls_labels_dict[name]==1]
#     if len(gt_labels)!=len(store_labels) or (gt_labels!=store_labels).max():
#         print("{}\tgt:{}\tstore:{}".format(name, gt_labels, store_labels))

'''
    用plt可视化多张图片
'''
# import os
# import matplotlib.pyplot as plt

# pic_dir = "/usr/volume/WSSS/VOCdevkit/VOC2012/JPEGImages"
# nums=90
# pic_per_line = 3
# pic_name_list = []
# pic_name_list = os.listdir(pic_dir)[:90]
# # figure, axes = plt.subplots(int(nums/pic_per_line), pic_per_line, figsize=())
# plt.figure(figsize=(12,90))
# for i in range(int(nums/pic_per_line)):
#     for j in range(pic_per_line):
#         plt.subplot(int(nums/pic_per_line), pic_per_line, i*pic_per_line+j+1)
#         plt.imshow(plt.imread(os.path.join(pic_dir,pic_name_list[i*pic_per_line+j])))
#         plt.xticks([])
#         plt.yticks([])
# plt.savefig('./test.jpg', bbox_inches='tight')

'''
    获取某一类对应的图片
'''
# import numpy as np
# import random

# classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
#             'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 
#             'potted plant', 'sheep', 'sofa', 'train', 'tv_monitor' ]

# label_file = '/usr/volume/WSSS/WSSS_PML/voc12/cls_labels.npy'
# labels = np.load(label_file, allow_pickle=True)
# labels_dict = labels.item()
# imgs_dict = dict()
# for i in range(20):
#     imgs_dict.setdefault(i, list())
# for item in labels_dict:
#     imgs_dict[np.argmax(labels_dict[item])].append(item)

# selected_nums = 10 
# random.seed(1234)
# for i in range(19,20):
#     tlen = len(imgs_dict[i])
#     idxs = random.sample(range(tlen), selected_nums)
#     save_file = '/usr/volume/WSSS/WSSS_PML/voc12/visualization/tensorboard_visualize_featuremap_img_{}.txt'.format(classes[i])
#     with open(save_file, mode='w') as f:
#         for idx in idxs:
#             f.write('/JPEGImages/{}.jpg /SegmentationClassAug/{}.png\n'.format(imgs_dict[i][idx], imgs_dict[i][idx]))

'''
    修改网络权重输出格式
'''
# import os
# import math
# import numpy
# in_file_path = "./weights-patch_weight-0.05-4ep.txt"
# out_file_path = "./test.txt"

# # with open(in_file_path, mode='r') as in_file:
# #     read_data = []
# #     for line in f.readlines():
# #         read_data.append(line.strip())
# classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
#             'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 
#             'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor' ]

# read_data = numpy.loadtxt(in_file_path)

# nums_per_line = 10
# with open(out_file_path, mode='w') as out_file:
#     for i in range(20):
#         out_file.write('class {}:{}\n'.format(i+1,classes[i]))
#         for j in range(math.ceil(len(read_data[i])/nums_per_line)):
#             if (j+1)*nums_per_line<len(read_data[i]):
#                 cur_data = read_data[i][j*nums_per_line:(j+1)*nums_per_line]
#             else:
#                 cur_data = read_data[i][j*nums_per_line:]
#             out_file.write(str(cur_data))

    
