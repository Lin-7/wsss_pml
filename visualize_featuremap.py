
import importlib
import numpy as np
import copy
from PIL import Image
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.models as models
import torchvision.transforms as transforms

import voc12.data
from tool import pyutils, imutils, torchutils, visualization

writer = SummaryWriter(log_dir='./visualize_featuremap', comment='this is a comment')

# 加载模型并载入参数
network = "network.resnet38_cls_ser_jointly_revised_seperatable"
weights = "/usr/volume/WSSS/wsss_pml/result/e5-patch_weight0.05/saved_checkpoints/patch_weight-0.05/4ep.pth"
model = getattr(importlib.import_module(network), 'Net')()
model.load_state_dict(torch.load(weights))


# # 获取模型中fc8_层的卷积核参数（20*256）
# fc8_weights = model.fc8_.state_dict()['weight'].squeeze()
# fc8_weights = np.array(fc8_weights)
# np.savetxt('weights-patch_weight-0.05-4ep.txt', fc8_weights)

# 加载数据
imgs_list_path = "/usr/volume/WSSS/wsss_pml/voc12/tensorboard_visualize_featuremap_img.txt"
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

# 注册hook
fmap_dict = dict()
fmap_dict_sample = dict()
n = 0
# 钩子函数
def hook_func(m, i, o):
    # print(m)
    key_name = str(m.weight.shape)
    fmap_dict[key_name].append(o)

# 初始化结果字典并将钩子函数勾到每一个卷积层上
for name, sub_module in model.named_modules():  # named_modules()返回网络的子网络层及其名称 
    if isinstance(sub_module, nn.Conv2d):
        n += 1
        key_name = str(sub_module.weight.shape)
        # key_name = 'Conv_'+str(n)
        # Python 字典 setdefault() 函数和 get()方法 类似, 如果键不存在于字典中，将会添加键并将值设为默认值。
        fmap_dict_sample.setdefault(key_name, list())
        # print(fmap_dict,'\n')
        if '.' in name:
            n1, n2 = name.split(".")
            model._modules[n1]._modules[n2].register_forward_hook(hook_func)
        else:
            model._modules[name].register_forward_hook(hook_func)
# fmap_dict_sample = fmap_dict.copy()

# 前向传播得到所有卷积层的输出,保存所有特征图
for iter, pack in enumerate(tensorboard_img_loader):
    fmap_dict = copy.deepcopy(fmap_dict_sample)
    tensorboard_img = pack[1]   # bchw
    tensorboard_label = pack[2].cuda(non_blocking=True)
    tensorboard_label = tensorboard_label.unsqueeze(2).unsqueeze(3)

    cam = model(tensorboard_img)
    # TODO: 先不叠加原图看效果
    for layer_name, fmap_list in fmap_dict.items():
        fmap = fmap_list[0]   # 同样大小的卷积核叠加多个，只取第一个的输出
        # print(fmap.shape)
        fmap.transpose_(0, 1)
        # print(fmap.shape)

        nrow = int(np.sqrt(fmap.shape[0]))
        # if layer_name == 'torch.Size([512, 512, 3, 3])':
        fmap = F.interpolate(fmap, size=[112, 112], mode="bilinear")
        
        fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
        print(type(fmap_grid),fmap_grid.shape)
        writer.add_image('feature map in {}'.format(layer_name), fmap_grid, iter)

