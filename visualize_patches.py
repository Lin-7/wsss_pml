import importlib
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import voc12.data
from tool import imutils 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 加载模型并载入参数
network = "network.resnet38_cls_ser_jointly_revised_seperatable"
filename = "e5-patch_weight0.05-fg-patchscale4-ming>0"
model_root = f"/usr/volume/WSSS/WSSS_PML/result/{filename}/"
weights = model_root + f"saved_checkpoints/4ep.pth"
model = getattr(importlib.import_module(network), 'Net')()
model.load_state_dict(torch.load(weights))

# 加载数据
imgs_list_path = "/usr/volume/WSSS/WSSS_PML/voc12/train_aug.txt"
voc12_root="/usr/volume/WSSS/VOCdevkit/VOC2012"
num_workers = 12
batch_size = 4
# tensorboard_dataset = voc12.data.VOC12ClsDataset(imgs_list_path, voc12_root=voc12_root,
#                                                     transform=transforms.Compose([
#                                                         np.asarray,
#                                                         model.normalize,
#                                                         imutils.CenterCrop(500),
#                                                         imutils.HWC_to_CHW,
#                                                         torch.from_numpy
#                                                     ]))
# tensorboard_img_loader = DataLoader(tensorboard_dataset,
#                                         shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
train_dataset = voc12.data.VOC12ClsDataset(imgs_list_path, voc12_root=voc12_root,
                                            transform=transforms.Compose([
                                                imutils.RandomResizeLong(448, 768),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                        hue=0.1),
                                                np.asarray,
                                                model.normalize,
                                                imutils.RandomCrop(448),
                                                imutils.HWC_to_CHW,
                                                torch.from_numpy
                                            ]))

train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True,
                                worker_init_fn=worker_init_fn)

model=torch.nn.DataParallel(model).cuda()

# 获取所有图片的patches以及对应的标签
patches = []
patch_labels = []
patch_nums = [4]
for iter, pack in tqdm(enumerate(train_data_loader)):

    name = pack[0]
    img = pack[1]
    label = pack[2].cuda(non_blocking=True)
    label = label.unsqueeze(2).unsqueeze(3)
    raw_H = pack[3]
    raw_W = pack[4]

    roi_cls_pooled, roi_label_list = model(x=img, label=label, patches=True, patch_nums=patch_nums)
    patches.extend(roi_cls_pooled)
    patch_labels.extend(roi_label_list)
    
patches = np.concatenate(patches, axis=0)
patch_labels = np.concatenate(patch_labels, axis=0)

# tSNE降维以及可视化
tsne = TSNE(n_components=2, learning_rate='auto').fit_transform(patches)
plt.figure(figsize=(30,30), dpi=80)
# plt.scatter(tsne[:,0], tsne[:, 1], c=patch_labels)
scatter = plt.scatter(tsne[:,0], tsne[:, 1], c=patch_labels, cmap=plt.cm.Spectral)
# plt.show()
# plt.colorbar()
plt.legend(handles=scatter.legend_elements(num=None)[0], labels=[f'{i}' for i in range(1,21)], loc='best')
plt.savefig(model_root + 'visualize_patches-s.jpg')

plt.figure(figsize=(20,20), dpi=80)
# plt.scatter(tsne[:,0], tsne[:, 1], c=patch_labels)
scatter = plt.scatter(tsne[:,0], tsne[:, 1], c=patch_labels, cmap=plt.cm.Spectral)
# plt.show()
# plt.colorbar()
plt.legend(handles=scatter.legend_elements(num=None)[0], labels=[f'{i}' for i in range(1,21)], loc='best')
plt.savefig(model_root + 'visualize_patches-g.jpg')