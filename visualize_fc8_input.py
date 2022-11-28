import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'wsodlib')
add_path(lib_path)
# end WSOD


import os 
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as transforms

import voc12.data
from tool import imutils, visualization
import importlib

classname = 'person'
selected_fms = {
    'person':[1,29,42,54,77,78,80,109,121,131,141,164,214,229],
}
logdir = f'./visualize_featuremap/last_layer/{classname}'
if not os.path.exists(logdir):
    os.makedirs(logdir)
writer = SummaryWriter(log_dir=logdir, comment='this is a comment')

# 加载模型并载入参数
network = "network.resnet38_cls_ser_jointly_revised_seperatable"
weights = "/usr/volume/WSSS/WSSS_PML/result/e5-patch_weight0.05/saved_checkpoints/patch_weight-0.05/4ep.pth"
model = getattr(importlib.import_module(network), 'Net')()
model.load_state_dict(torch.load(weights))

# # 获取模型中fc8_层的卷积核参数（20*256）
# fc8_weights = model.fc8_.state_dict()['weight'].squeeze()
# fc8_weights = np.array(fc8_weights)
# np.savetxt('weights-patch_weight-0.05-4ep.txt', fc8_weights, fmt='%.4f')

# 加载数据
imgs_list_path = f"/usr/volume/WSSS/WSSS_PML/voc12/visualization/tensorboard_visualize_featuremap_img_{classname}.txt"
voc12_root="/usr/volume/WSSS/VOCdevkit/VOC2012"
num_workers = 12
tensorboard_dataset = voc12.data.VOC12ClsDataset(imgs_list_path, voc12_root=voc12_root,
                                                    transform=transforms.Compose([
                                                        np.asarray,
                                                        model.normalize,
                                                        imutils.CenterCrop(500),
                                                        imutils.HWC_to_CHW,
                                                        torch.from_numpy
                                                    ]))
tensorboard_img_loader = DataLoader(tensorboard_dataset,
                                        shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

# TODO
# 4 添加代码：只显示最后一卷积层的特征(原始特征)，同时与原图进行叠加
# 1 imglist 按类别分,文件中按类别选择样本进行特征图可视化,tb中iter标识不同的图片

# 对每张图片进行
#   1 前向传播得到对应的特征图
#   2 保存特征图
#       a 直接可视化特征图
#       b 用可视化cam的方法将特征图叠加到原图上进行显示
all_results = []
nrow = 5
times = 1
for iter, pack in enumerate(tensorboard_img_loader):
    print(iter)
    tensorboard_img = pack[1]   # bchw  归一化后的结果，如果要可视化需要恢复到0-255的范围
    _, c, h, w = tensorboard_img.size()
    fmap = model(tensorboard_img, featuremap=True)
    fmap.transpose_(0,1)

    # 处理原图
    img_8 = tensorboard_img[0].numpy().transpose((1, 2, 0))
    img_8 = np.ascontiguousarray(img_8)  # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_8[:, :, 0] = (img_8[:, :, 0] * std[0] + mean[0]) * 255
    img_8[:, :, 1] = (img_8[:, :, 1] * std[1] + mean[1]) * 255
    img_8[:, :, 2] = (img_8[:, :, 2] * std[2] + mean[2]) * 255
    img_8[img_8 > 255] = 255
    img_8[img_8 < 0] = 0
    img_8 = img_8.astype(np.uint8)
    image = img_8.transpose((2, 0, 1))

    # # 1 不叠加原图直接显示
    # # 256个特征图分4*8*8显示
    # for i in range(times):
    #     current_maps = fmap[i*each:(i+1)*each]
    #     # normalize:图像归一化到0-1之间; scale_each:对每张图片各自进行归一化
    #     fmap_grid = vutils.make_grid(current_maps, normalize=True, scale_each=True, nrow=nrow)
    #     writer.add_image('feature map {}'.format(i), fmap_grid, iter)

    each = int(fmap.size(0)/times)
    # 2 规范化处理并叠加原图后显示
    for i in range(times):
        current_maps = fmap[i*each:(i+1)*each]
        
        # 处理特征图
        # 双线性插值到原图大小  TODO 是否需要考虑channel维度的指定
        current_maps = F.interpolate(current_maps, (h, w), mode='bilinear')     # TODO 原图是500*500 特征图是63*63 直接缩放成原图大小可能幅度有点大
        # 像素值归一化
        current_maps = visualization.max_norm(current_maps, 'torch')

        results = visualization.ColorCAM(current_maps.squeeze().detach().numpy(), image)
        all_results.append(results)

        # showimg_grid = vutils.make_grid([torch.tensor(t) for t in results], nrow=nrow)
        # writer.add_image('feature map and image {}'.format(i), showimg_grid, iter)

# 3 只输出部分指定的特征图
all_results = torch.tensor(np.stack(all_results, axis=0))
for idx in selected_fms[classname]:
    tempshowimg = all_results[:,idx,:,:,:].squeeze()
    tempshowimg_grid = vutils.make_grid(tempshowimg, nrow=nrow)
    writer.add_image('selected feature map', tempshowimg_grid, idx)
