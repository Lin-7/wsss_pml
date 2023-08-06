#coding:utf-8
'''
    将原来的train_cls_loc_jointly.py infer_cls_pml.py整合在一起，
    这样评估的时候不用再次装载模型，可以节省显存，从而可以设置大一点的batchsize
'''
import os
import torch
import random
import numpy as np

import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

torch.set_printoptions(profile="full")
torch.manual_seed(7) # cpu
torch.cuda.manual_seed(7) #gpu
np.random.seed(7) #numpy
random.seed(7) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
print(f"Random seed is set as 7")

import cv2
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import argparse
import importlib
import time
import shutil
from tqdm import tqdm
from PIL import Image

from evaluation import eval
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import glo

# seed = pyutils.seed_everything()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']

def worker_init_fn(worker_id):
    # 重新设置dataloader中每个worker对应的np和random的随机种子
    np.random.seed(1+worker_id)
    # random.seed(worker_seed)

# g = torch.Generator()
# # 设置样本shuffle随机种子，作为DataLoader的参数
# g.manual_seed(0)


if __name__ == '__main__':
    '''
        !!! 训练和评估是一起执行的，因此训练的时候确定batch_size时要给评估留时间
        !! batch_size和num_worker都要调整
    '''

    parser = argparse.ArgumentParser()
    # 机器和环境的不同，会差一两个点
    parser.add_argument("--batch_size", default=4, type=int)   # 10/12   一个gpu：8×6√  两个gpu：10
    parser.add_argument("--max_epoches", default=3, type=int)   # 根据机器去修改
    parser.add_argument("--network", default="network.resnet38_cls_moco", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_workers_infer", default=12, type=int)   # torch.utils.data.dataloader提示建议创建12个workers
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    # 权重
    parser.add_argument("--weights", default="/usr/volume/WSSS/weights_released/res38_cls.pth", type=str)
    # parser.add_argument("--weights", default="/usr/volume/WSSS/weights_released/resnet38_cls_ser_0.3.pth", type=str)

    # 数据集位置
    parser.add_argument("--voc12_root", default="/usr/volume/WSSS/VOCdevkit/VOC2012", type=str)

    # 数据集划分文件
    # parser.add_argument("--train_list", "-tr", default="/usr/volume/WSSS/WSSS_PML/voc12/train_voc12_mini_fortest.txt", type=str)   # 测试用的小批的训练数据
    parser.add_argument("--train_list", "-tr", default="/usr/volume/WSSS/WSSS_PML/voc12/train_aug.txt", type=str)
    parser.add_argument("--infer_list", default=f"/usr/volume/WSSS/WSSS_PML/voc12/val_voc12.txt", type=str)  # 跟phase指定的值要一致
    # parser.add_argument("--infer_list", default=f"/usr/volume/WSSS/WSSS_PML/voc12/train_voc12_mini_fortest.txt", type=str)  # 测试用的小批的训练数据
    parser.add_argument("--tensorboard_img", default="/usr/volume/WSSS/WSSS_PML/voc12/tensorborad_img.txt", type=str)  # 用来生成tf展示的图片

    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--optimizer", default='poly', type=str)

    # patch相关的参数
    parser.add_argument("--patch_gen", default="randompatch", type=str)   # 4patch randompatch contrastivepatch
    parser.add_argument("--ccrop_alpha", default=0.7, type=float)
    parser.add_argument("--patch_select_close", default=False, type=bool)   # 不选择patches
    parser.add_argument("--patch_select_cri", default="fgratio", type=str)   # fgratio confid fgAndconfid random
    parser.add_argument("--patch_select_ratio", default=0.4, type=float)   # 0.3 0.4 0.5
    parser.add_argument("--patch_select_part_fg", default="mid", type=str)   # front mid back
    parser.add_argument("--patch_select_part_confid", default="front", type=str)   # front mid back
    # parser.add_argument("--patch_select_checksimi", default=True, type=bool)   # 选择patches的时候考虑内容相似性
    # parser.add_argument("--patch_select_checksimi_thres", default=0.9, type=float)
    parser.add_argument("--proposal_padding", default=0, type=float)
    parser.add_argument("--use_queue", default=False, type=bool)
    parser.add_argument("--queuesize", default=1000, type=int)
    parser.add_argument("--model_update_m", default=0.999, type=float)
    parser.add_argument("--is_hard_negative", default=False, type=bool)
    parser.add_argument("--bghard", default=False, type=bool)
    parser.add_argument("--patch_loss_weight", default=0.05, type=float)

    parser.add_argument("--session_name", default="0427-randomGen-fgmid0.4-noQ-noM-nohard", type=str)         # train val test
    # parser.add_argument("--session_name", default="e3-patch_weight0.05-all-randompatch-fgmid0.5-Sp0.3-noNp10-noNMS-11", type=str)         # train val test
    # parser.add_argument("--session_name", default="e3-patch_weight0.05-all-padding0.25-10patch_randomstart-fgmid0.4-seed7", type=str)         # train val test
    # patch_loss_weight = 0.05
    parser.add_argument("--tblog", default="saved_checkpoints", type=str)
    
    # 评估参数
    phase = "val"               # 要和infer_list 一致（infer_list指定的文件有包括图片路径和类别，phase指定的文件只包含名称，用于找到指定的cam）
    bg_thresh=[0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30]
    crf_alpha = [4, 16, 24, 28, 32]
    # 指定评估结果输出的文件
    # parser.add_argument("--out_cam", default="./out_cam_val", type=str)   # 保存每张图片中的每个目标物体类别的CAM
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--out_crf", default=None, type=str)  # 保存条件随机场修正后的out_cam
    # parser.add_argument("--out_crf", default="./out_crf", type=str)
    # parser.add_argument("--out_cam_pred", default="./out_cam_pred_val", type=str)  # 保存每张图片中包括背景类的所有类别的CAM
    # parser.add_argument("--log_infer_cls", default=f"/usr/volume/WSSS/WSSS_PML/log_CAM_{phase}.txt", type=str)

    args = parser.parse_args()
    args.interpolate_mode = 'bilinear'  # bicubic, nearest

    # 存放结果的根目录
    out_root = f"/usr/volume/WSSS/WSSS_PML/result/{args.session_name}/"
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    # 存放模型和模型评估结果的目录
    log_root = out_root + f"{args.tblog}/"
    if os.path.exists(log_root):
        shutil.rmtree(log_root)
    os.makedirs(log_root, exist_ok=True)

    # 存放模型对验证集基于cam的预测结果的目录
    out_cam_pred_dir = out_root + "out_cam_pre"
    if os.path.exists(out_cam_pred_dir):
        shutil.rmtree(out_cam_pred_dir)
    os.makedirs(out_cam_pred_dir, exist_ok=True)

    # 存放验证集对应的cams的目录
    out_cam = out_root + "out_cams"
    if os.path.exists(out_cam):
        shutil.rmtree(out_cam)
    os.makedirs(out_cam, exist_ok=True)

    # 存放可视化图的目录
    args.visualize_patch_dir = out_root + 'visualization'
    if os.path.exists(args.visualize_patch_dir):
        shutil.rmtree(args.visualize_patch_dir)
    os.makedirs(args.visualize_patch_dir, exist_ok=True)


    # 复制主要运行文件
    copy_files_list = ['/usr/volume/WSSS/WSSS_PML/train_cls_loc_jointly_new_dp.py', 
                        '/usr/volume/WSSS/WSSS_PML/network/resnet38_cls_moco.py',
                        '/usr/volume/WSSS/WSSS_PML/network/resnet38d_moco.py',
                        '/usr/volume/WSSS/WSSS_PML/tool/RoiPooling_Jointly.py',
                        '/usr/volume/WSSS/WSSS_PML/tool/pyutils.py']
    for copy_file in copy_files_list: 
        shutil.copy(copy_file, out_root)

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.init_roi_pooling_method(args)

    # 存放模型训练过程数据记录的目录（参数指定的是tensorboard文件的路径）
    tblogger = SummaryWriter(out_root + args.tblog+'log')
    # 训练日志文件
    # 重写并替换了sys.stdout类，重新指定了输出的位置（同时写到终端和指定的文件中）
    pyutils.Logger(out_root + args.session_name + '.log')

    print(vars(args))

    # dataset
    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                                                   imutils.RandomResizeLong(448, 768),   # 随机将长边resize到448-768之间的值，短边自适应
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                          hue=0.1),
                                                   np.asarray,
                                                   imutils.Normalize(),
                                                   imutils.RandomCrop(args.crop_size),   # 随机裁剪出448*448
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)

    # train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
    #                                shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
    #                                worker_init_fn=worker_init_fn,generator=g)


    max_step = (len(train_dataset) // args.batch_size)*args.max_epoches

    tensorboard_dataset = voc12.data.VOC12ClsDataset(args.tensorboard_img, voc12_root=args.voc12_root,
                                                     transform=transforms.Compose([
                                                         np.asarray,
                                                         imutils.Normalize(),
                                                         imutils.CenterCrop(500),
                                                         imutils.HWC_to_CHW,
                                                         torch.from_numpy
                                                     ]))
    tensorboard_img_loader = DataLoader(tensorboard_dataset,
                                        shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                scales=[1, 0.5, 1.5, 2.0],
                                                inter_transform=torchvision.transforms.Compose(
                                                    [np.asarray,
                                                    imutils.Normalize(),
                                                    imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers_infer, pin_memory=True)

    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls_ser_jointly"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.encoder_q.load_state_dict(weights_dict, strict=False)
    model.momentum_encoder_init()

    param_groups = model.get_parameter_groups()
    if args.optimizer=='poly':
        optimizer = torchutils.PolyOptimizer([
            {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
            {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
            {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
        ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    elif args.optimizer=='adam':
        optimizer=torchutils.Adam([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    model=torch.nn.DataParallel(model).cuda()

    model.train()

    avg_meter = pyutils.AverageMeter('loss')
    avg_meter1 = pyutils.AverageMeter('loss_cls')
    avg_meter2 = pyutils.AverageMeter('loss_patch')

    timer = pyutils.Timer("Session started: ")

    loss_list=[]
    validation_set_CAM_mIoU=[]
    val_loss_list = []
    train_set_CAM_mIoU = []
    val_multi_mIoU = []
    train_multi_mIoU = []

    global_step = 0
    patch_num = 0
    max_step_small = 0

    is_opti = True

    is_need_load_proposal_data=False
    is_init_opti = False
    train_small_dataset=None
    train_small_data_loader=None
    optimizer_patch_cls = None

    bounding_box_dict = np.load('voc12/bounding_box.npy', allow_pickle=True).item()
    rw, rh = [448, 448]
    
    # training
    patchnums, npatchnums, ppatchnums = [], [], []
    glo._init()
    glo.set_value('label_queue', torch.randn(args.queuesize, 1))
    glo.set_value('Fbg_queue', torch.randn(args.queuesize, 256))
    glo.set_value('F_queue', torch.randn(args.queuesize, 256))
    for ep in range(args.max_epoches):
        itr = ep + 1
        # log the images at the beginning of each epoch
        
        '''
        for iter, pack in enumerate(tensorboard_img_loader):
            tensorboard_img = pack[1]
            tensorboard_label = pack[2].cuda(non_blocking=True)
            tensorboard_label = tensorboard_label.unsqueeze(2).unsqueeze(3)
            N, C, H, W = tensorboard_img.size()
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

            input_img = img_8.transpose((2, 0, 1))            # tensorboard_img[0] → img_8 → input_img(规范化且内存连续)
            h = H // 4
            w = W // 4
            model.eval()
            with torch.no_grad():
                cam = model(x=tensorboard_img,args=args)
            model.train()
            # Down samples the input to the given size
            p = F.interpolate(cam, (h, w), mode=args.interpolate_mode, align_corners=False)[0].detach().cpu().numpy()     
            bg_score = np.zeros((1, h, w), np.float32)
            p = np.concatenate((bg_score, p), axis=0)  # 追加背景cam
            bg_label = np.ones((1, 1, 1), np.float32)
            l = tensorboard_label[0].detach().cpu().numpy()
            l = np.concatenate((bg_label, l), axis=0)   # 追加背景标签
            # Donotunderstand: w, h ？ 
            image = cv2.resize(img_8, (w, h), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))   # chw
            # 应该是可视化图片
            CLS, CAM, CLS_crf, CAM_crf = visualization.generate_vis(p, l, image,
                                                                    func_label2color=visualization.VOClabel2colormap)
            tblogger.add_image('Image_' + str(iter), input_img, itr)
            tblogger.add_image('CLS_' + str(iter), CLS, itr)
            tblogger.add_image('CLS_crf' + str(iter), CLS_crf, itr)
            tblogger.add_images('CAM_' + str(iter), CAM, itr)
        '''

        patches = []
        p_labels = []
        p_masks = []
        # visualize_iter_idxs = np.random.choice(range(ep*len(train_dataset), (ep+1)*len(train_dataset)), 10, replace=False) # 可视化前随机5个iter
        # visualize_iter_idxs = range(0, len(train_dataset))[:10] # 可视化前10个iter
        visualize_iter_idxs = [] # 测试阶段先不用可视化
        for iter, pack in tqdm(enumerate(train_data_loader)):

            name = pack[0]
            img = pack[1]
            label = pack[2].cuda(non_blocking=True)
            label = label.unsqueeze(2).unsqueeze(3)
            raw_H = pack[3]
            raw_W = pack[4]
            pack3 = []
            param = [rw, rh, raw_W, raw_H, patch_num]
            bounding_box = []
            
            for i in range(args.batch_size):
                bounding_box.append(bounding_box_dict[str(name[i])])

            optimizer.zero_grad()

            if iter in visualize_iter_idxs:
                # visualize patches
                loss_cls, loss_patch, patch_embs, patch_labels, patch_mask, queues_info \
                 = model(x=img, label=label, bounding_box=bounding_box, param=param, \
                    is_patch_metric=True, is_sse=False, epoch_iter=f"{ep}_{iter}", \
                        img_names=name, args=args)
            else:
                loss_cls, loss_patch, patch_embs, patch_labels, patch_mask, queues_info \
                 = model(x=img, label=label, bounding_box=bounding_box, param=param, \
                    is_patch_metric=True, is_sse=False, img_names=name, \
                        args=args)

            loss_cls=loss_cls.mean()

            loss_patch=loss_patch.mean()

            # # baseline:no metric learning
            # loss_cls = model(x=img, label=label, param=param, is_patch_metric=False, is_sse=False)[0]
            # loss_cls=loss_cls.mean()
            # loss_patch=torch.tensor(0)
                # loss=loss_cls+loss_patch/20
                ### 2022
            # 原本的模型的loss_cls已经训练得很好了,而我们新加的loss_patch的数组较大,所以/10让其变小
            # 不然数值太大,甚至大了一个数量级会让模型太过专注于这一部分,
            # 但这属于一个multi task的训练, 我们的整个框架依赖于模型要有一个比较小的loss_cls, 才能持续地输出准确的cam
            # 因此要将它们两reweight到一个数量级,让模型同时去关注这两个点
            # 有多个loss的: 最基础的--reweight到同一个数量级,但是具体的表现或者说数值还是得通过实验结果去看(10,20,30都是一个数量级嘛,很多情况)
            loss = loss_cls + loss_patch * args.patch_loss_weight
            # loss = loss_cls
            avg_meter2.add({'loss_patch': loss_patch.item()})
            avg_meter.add({'loss': loss.item()})
            avg_meter1.add({'loss_cls': loss_cls.item()})

            # loss.requires_grad_(True)
            # loss.backward(retain_graph=True)
            loss.backward()

            optimizer.step()

            global_step+=1

            q_features, q_nfg_features, q_labels = queues_info
            q_features = q_features.detach().cpu()
            q_nfg_features = q_nfg_features.cpu()
            q_labels = q_labels.cpu().unsqueeze(1)

            label_queue = glo.get_value('label_queue')
            Fbg_queue = glo.get_value('Fbg_queue')
            F_queue = glo.get_value('F_queue')
            cur_patches_size = q_features.size(0)
            F_queue = torch.cat((F_queue, q_features), dim=0)[cur_patches_size:,:]
            Fbg_queue = torch.cat((Fbg_queue, q_nfg_features), dim=0)[cur_patches_size:,:]
            label_queue = torch.cat((label_queue, q_labels), dim=0)[cur_patches_size:,:]
            glo.set_value('label_queue', label_queue)
            glo.set_value('Fbg_queue', Fbg_queue)
            glo.set_value('F_queue', F_queue)

            patches.extend(patch_embs.detach().cpu())
            p_labels.extend(patch_labels.cpu())
            p_masks.extend(patch_mask.cpu())

            # print
            if (global_step-1)%10 == 0:
                timer.update_progress(global_step / max_step)

                a=avg_meter.get('loss')
                loss_list.append(avg_meter.get('loss'))

                print('Iter:%5d/%5d' % (global_step - 1, max_step),
                      'Loss: %.4f' % (avg_meter.get('loss')),
                      'Loss_cls: %.4f:'%(avg_meter1.get('loss_cls')),
                      'Loss_patch: %.4f:' % (avg_meter2.get('loss_patch')),
                      'imps:%.3f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.6f' % (optimizer.param_groups[0]['lr']), flush=True)
                avg_meter.pop()
            
        print(f"epoch{ep} end!!!!!!!!!!!!!!!!!!!")

        '''
        # ==== 附加内容:统计用于构造triplet的patches总数以及没有负正对的patches总数 ======
        dir1 = "/usr/volume/WSSS/WSSS_PML/somefiles/patchnum_0.txt"
        dir2= "/usr/volume/WSSS/WSSS_PML/somefiles/patchnum_1.txt"
        newname1 = f"/usr/volume/WSSS/WSSS_PML/somefiles/patchnum_0-epoch{ep}.txt"
        newname2 = f"/usr/volume/WSSS/WSSS_PML/somefiles/patchnum_1-epoch{ep}.txt"
        num, num1, num2 = 0, 0, 0
        with open(dir1, 'r') as f:
            for line in f.readlines():
                tnum, tnum1, tnum2 = line.strip('\n').split()
                num += int(tnum)
                num1 += int(tnum1)
                num2 += int(tnum2)
        with open(dir2, 'r') as f:
            for line in f.readlines():
                tnum, tnum1, tnum2 = line.strip('\n').split()
                num += int(tnum)
                num1 += int(tnum1)
                num2 += int(tnum2)
        os.rename(dir1,newname1)
        os.rename(dir2,newname2)
        patchnums.append(num)
        npatchnums.append(num1)
        ppatchnums.append(num2)
        # ==== 附加内容:统计用于构造triplet的patches总数以及没有负正对的patches总数 ======
        '''
        ''' 
        # ==== 附加内容:统计随机生成的10个patch的分布情况 ======
        # dir1 = "/usr/volume/WSSS/WSSS_PML/distances_0.txt"
        # dir2= "/usr/volume/WSSS/WSSS_PML/distances_1.txt"
        # newname1 = f"/usr/volume/WSSS/WSSS_PML/distances_0-epoch{ep}.txt"
        # newname2 = f"/usr/volume/WSSS/WSSS_PML/distances_1-epoch{ep}.txt"
        # dists = []
        # with open(dir1, 'r') as f:
        #     for line in f.readlines():
        #         dists.append(float(line.strip('\n')))
        # with open(dir2, 'r') as f:
        #     for line in f.readlines():
        #         dists.append(float(line.strip('\n')))
        # os.rename(dir1,newname1)
        # os.rename(dir2,newname2)
        # from matplotlib import pyplot as plt
        # d = 0.1
        # num_bins = 10
        # plt.figure(figsize=(20,8), dpi=80)
        # plt.hist(dists, bins=10)
        # xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # plt.xticks(xticks)
        # plt.grid(alpha=0.4)
        # plt.savefig(f"/usr/volume/WSSS/WSSS_PML/distances_hist-epoch{ep}.jpg")
        # ==== 附加内容:统计随机生成的10个patch的分布情况 ======
        '''

        if args.optimizer=='adam':
            optimizer.adam_turn_step()

        # visualize_start = time.time()
        # # 可视化所有patch的特征分布，所选择的patch，构造triplet的patch
        # visualization.visualize_patch(patches=torch.stack(patches).numpy(), patch_labels=torch.stack(p_labels).numpy(), patch_mask=torch.stack(p_masks).numpy(), save_dir=args.visualize_patch_dir, epoch=ep)
        # print("time for visualization:{:.2f}s".format(time.time()-visualize_start))

        # 每个epoch保存模型
        model_saved_root=log_root
        os.makedirs(model_saved_root, exist_ok=True)
        model_saved_dir = os.path.join(model_saved_root, f"{ep}ep.pth")
        # torch.save(model.module.state_dict(),model_saved_dir)

        avg_meter.pop()

        # evaluation
        loss_dict = {'loss': loss_list[-1]}
        tblogger.add_scalars('cls_loss', loss_dict, itr)
        tblogger.add_scalar('cls_lr', optimizer.param_groups[0]['lr'], itr)

        # eval
        # os.system(f"/opt/conda/envs/torch-python37/bin/python infer_cls_pml.py --log_infer_cls {result_saved_dir}.txt --weights {model_saved_dir} --out_cam_pred ./out_cam_ser_{args.session_name}")
        result_saved_dir=os.path.join(f"{log_root}/log_txt/", f"{args.session_name}_{ep}")
        os.makedirs(f"{log_root}/log_txt/", exist_ok=True)
        args.log_infer_cls = result_saved_dir
        result_saved_dir_detail=os.path.join(f"{log_root}/log_txt/", f"detail_{args.session_name}_{ep}")
        args.log_infer_cls_detail = result_saved_dir_detail

        model.eval()
        # 用当前的模型计算出cam和crf修正后的cam，同时对cam进行评估，评估结果输出到log_infer_cls中
        # 由于cam很多，因此每个epoch生成的cam会覆盖之前的
        # makedir stuff ================================================================
        if out_cam_pred_dir is not None:   # 
            args.out_cam_pred = f"{out_cam_pred_dir}/epoch{ep}"
            if os.path.exists(args.out_cam_pred):
                shutil.rmtree(args.out_cam_pred)
            if not os.path.exists(args.out_cam_pred):
                os.makedirs(args.out_cam_pred)
            for background_threshold in bg_thresh:
                os.makedirs(f"{args.out_cam_pred}/{background_threshold}", exist_ok=True)

        if out_cam is not None:
            # args.out_cam = f"{out_cam}/epoch{ep}"
            args.out_cam = f"{out_cam}"
            if os.path.exists(args.out_cam):
                shutil.rmtree(args.out_cam)
            if not os.path.exists(args.out_cam):
                os.makedirs(args.out_cam)

        if args.out_crf is not None:
            if os.path.exists(args.out_crf):
                shutil.rmtree(args.out_crf)
            for t in crf_alpha:
                folder = args.out_crf + ('_%.1f' % t)
                if not os.path.exists(folder):
                    os.makedirs(folder)
        # ==============================================================================

        n_gpus = torch.cuda.device_count()
        for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
            img_name = img_name[0]; label = label[0]

            img_path = voc12.data.get_img_path(img_name, args.voc12_root)
            orig_img = np.asarray(Image.open(img_path))
            orig_img_size = orig_img.shape[:2]

            def _work(i, img):
                with torch.no_grad():
                    with torch.cuda.device(i%n_gpus):
                        cam = model(x=img.cuda(), args=args)
                        # print(cam)
                        cam = F.relu(cam, inplace=True)
                        if args.interpolate_mode == "bilinear":
                            cam = F.interpolate(cam, orig_img_size, mode=args.interpolate_mode, align_corners=False)[0]
                        else:
                            cam = F.interpolate(cam, orig_img_size, mode=args.interpolate_mode)[0]
                        cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                        if i % 2 == 1:
                            cam = np.flip(cam, axis=-1)
                        return cam

            thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                                batch_size=8, prefetch_size=0, processes=args.num_workers_infer)

            cam_list = thread_pool.pop_results()

            sum_cam = np.sum(cam_list, axis=0)
            norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)   # 使得每张图片的每张cam的激活值位于[0,1]

            cam_dict = {}
            for i in range(20):
                if label[i] > 1e-5:
                    cam_dict[i] = norm_cam[i]

            # if args.out_cam is not None:  # 部分CAM
            #     np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

            if args.out_cam_pred is not None:   # 根据cams的预测结果
                for background_threshold in bg_thresh:
                    bg_score = [np.ones_like(norm_cam[0])*background_threshold]  
                    pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
                    cv2.imwrite(os.path.join(f"{args.out_cam_pred}/{background_threshold}", img_name + '.png'), pred.astype(np.uint8))

            def _crf_with_alpha(cam_dict, alpha):
                v = np.array(list(cam_dict.values()))
                bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
                bgcam_score = np.concatenate((bg_score, v), axis=0)
                crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

                n_crf_al = dict()

                n_crf_al[0] = crf_score[0]
                for i, key in enumerate(cam_dict.keys()):
                    n_crf_al[key+1] = crf_score[i+1]

                return n_crf_al

            if args.out_crf is not None:
                for t in crf_alpha:
                    crf = _crf_with_alpha(cam_dict, t)
                    folder = args.out_crf + ('_%.1f'%t)
                    # if not os.path.exists(folder):
                    #     os.makedirs(folder)
                    np.save(os.path.join(folder, img_name + '.npy'), crf)
            if iter%10==0:
                print(iter)

        for background_threshold in bg_thresh:
            if args.out_cam_pred is not None:
                print(f"background threshold is {background_threshold}")
                
                visualize_dir=args.visualize_patch_dir+f"/epoch{ep}"
                if not os.path.exists(visualize_dir):
                    os.mkdir(visualize_dir)
                visualize_dir=visualize_dir+f"/bg_thres_{background_threshold}"
                if not os.path.exists(visualize_dir):
                    os.mkdir(visualize_dir)

                eval(f"/usr/volume/WSSS/WSSS_PML/voc12/{phase}.txt", f"{args.out_cam_pred}/{background_threshold}", saved_txt=args.log_infer_cls, \
                    detail_txt=args.log_infer_cls_detail, visualize_dir=visualize_dir, cams_dir=args.out_cam, \
                        model_name=args.weights, bg_threshold=background_threshold)
                # 测试用的小批量数据
                # eval(f"/usr/volume/WSSS/WSSS_PML/voc12/train_mini_fortest.txt", f"{args.out_cam_pred}/{background_threshold}", saved_txt=args.log_infer_cls, \
                #     detail_txt=args.log_infer_cls_detail, visualize_dir=visualize_dir, cams_dir=args.out_cam, \
                #         model_name=args.weights, bg_threshold=background_threshold)

    '''
    from matplotlib import pyplot as plt
    width=0.3
    x = np.arange(3)
    x1 = x-width/2
    x2 = x1 + width
    x3 = x2 + width
    plt.bar(x1, patchnums, width=width, label="total")
    plt.bar(x2, npatchnums, width=width, label="NoNeg")
    plt.bar(x3, ppatchnums, width=width, label="NoPos")
    plt.ylabel("nums")
    for a,b in zip(x1, patchnums):
        plt.text(a, b, f"{b}", ha='center', va='bottom', fontsize=7)
    for a,b in zip(x2, npatchnums):
        plt.text(a, b, f"{b}", ha='center', va='bottom', fontsize=7)
    for a,b in zip(x3, ppatchnums):
        plt.text(a, b, f"{b}", ha='center', va='bottom', fontsize=7)
    plt.xticks(x, ["epoch0", "epoch1", "epoch2"])
    plt.savefig(f"/usr/volume/WSSS/WSSS_PML/patches-condition.jpg")

    print("Session finished:{}".format(time.ctime(time.time())))
    '''
