import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tool.RoiPooling_Jointly import RoiPooling
import network.resnet38d

import cv2
import torchvision
import random
import math


# begin WSOD config
from wsodlib.nets.resnet_v1 import resnetv1_fast
from wsodlib.layer_utils.generate_anchors import _whctrs
from wsodlib.model.bbox_transform import bbox_transform_inv, clip_boxes
# end WSOD

# from pytorch_metric_learning import miners, losses
# miner = miners.MultiSimilarityMiner()
# loss_func = losses.TripletMarginLoss()


class Net(network.resnet38d.Net):
    def __init__(self):
        super().__init__()

        # loss
        self.loss_cls = 0
        self.loss_patch_location = 0
        self.loss_patch_cls = 0
        self.loss=0

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8_ = nn.Conv2d(256, 20, 1, bias=False)


        torch.nn.init.xavier_uniform_(self.fc8_.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8_]

        # patch-based metric learning loss
        self.patch_based_metric_init()
        self.init_teacher('lbba_final.pth')


    def init_teacher(self, teacher_model):
        # WOSD teacher
        self.teacher = resnetv1_fast(num_layers=50)   # resnet50
        # Build the main computation graph
        self.teacher.create_architecture(num_classes=61, anchor_scales=(4, 8, 16, 32), tag='default')
        teacher_pre = torch.load(teacher_model)
        teacher_dict = self.teacher.state_dict()
        teacher_dict.update(teacher_pre)
        self.teacher.load_state_dict(teacher_dict)
        print('Loading pretrained teacher model weights from {:s}'.format(teacher_model))
        # self.teacher.cuda()
        self.teacher=torch.nn.DataParallel(self.teacher).cuda()
        self.teacher.eval()

    def patch_based_metric_init(self,):

        # 目标1：降低特征的维度，便于计算
        # 目标2：多了一个非线性层，给模型多一些参数来学习我们施加的patch-based metric learning
        self.downsample = nn.Conv2d(4096, 256, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.downsample.weight)
        self.from_scratch_layers.append(self.downsample)

        # 平均池化，得到CAM的bounding box后，要对这一个区域中的特征做一个平均池化
        # 1 先align box的位置
        # 2 对box内的feature map做一个平均池化
        self.roi_pooling = RoiPooling(mode="th")
        self.ranking_loss = nn.MarginRankingLoss(margin=28.0)
        self.ranking_loss_same_img = nn.MarginRankingLoss(margin=28.0)

    # 输入CAM图片和对应的类别
    # 返回bounding box的坐标和对应的类别
    def get_roi_index(self, cam, cls_label):
        '''
        For each image
        :param cam: 20 * W* H
        :param cls_label:
        :return:
        roi_index[N,4]
        4->[XMIN,YMIN,XMAX,YMAX]

        label=[N]

        limitation:
        '''
        bg_threshold=0.20
        iou_threshhold=0.2
        W,H = cam.size(1),cam.size(2)

        cam = F.relu(cam)
        cam=cam.mul(cls_label).cpu().numpy()

        # 规范化之后把背景类的cam补上
        norm_cam = cam / (np.max(cam, (1, 2), keepdims=True) + 1e-5)
        bg_score = [np.ones_like(norm_cam[0]) * bg_threshold]
        cam_predict = np.argmax(np.concatenate((bg_score, norm_cam)), 0)

        label = np.unique(cam_predict)
        label = label[1:]  # get the label except background
        # for each class

        bounding_box = {}
        bounding_scores = {}

        roi_index=[]
        label_list=[]

        for l in label:
            label_i = np.zeros((W,H))
            label_i[cam_predict == l] = 255
            label_i = np.uint8(label_i)
            contours, hier = cv2.findContours(label_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到CAM中的轮廓
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w < 9 or h < 9 or (h / w) > 4 or (w / h) > 4:  # filter too small bounding box
                    continue
                else:
                    xmin = x
                    ymin = y
                    xmax = x + w
                    ymax = y + h

                    bbox_region_cam = norm_cam[l-1, ymin:ymax, xmin:xmax]
                    bbox_score = np.average(bbox_region_cam)
                    if l not in bounding_box:
                        bounding_box[l] = list([[xmin, ymin, xmax, ymax]])
                        bounding_scores[l] = [bbox_score]
                    else:
                        bounding_box[l].append(list([xmin, ymin, xmax, ymax]))
                        bounding_scores[l].append(bbox_score)
            # NMS step
            if l in bounding_box:
                b = torch.from_numpy(np.array(bounding_box[l],np.double)).cuda()
                s = torch.from_numpy(np.array(bounding_scores[l],np.double)).cuda()
                bounding_box_index = torchvision.ops.nms(b,s,iou_threshhold)

                for i in bounding_box_index:
                    # generate new small dataset
                    xmin = bounding_box[l][i][0]
                    ymin = bounding_box[l][i][1]
                    xmax = bounding_box[l][i][2]
                    ymax = bounding_box[l][i][3]

                    roi_index.append(list([xmin, ymin, xmax, ymax]))
                    label_list.append(l)
                    if len(label_list)>=3:    # 每张图片，每个label最多返回三个轮廓
                        # print("2")
                        # return roi_index, label_list
                        break
        return roi_index, label_list

    # 可能是之前做消融留下的函数
    def get_roi_index_patch_same_image(self, cam, cls_label):
        '''
        For each image
        :param cam: 20 * W* H
        :param cls_label:
        :return:
        roi_index[N,4]
        4->[XMIN,YMIN,XMAX,YMAX]

        label=[N]

        limitation:
        '''
        bg_threshold=0.20
        iou_threshhold=0.2
        W,H = cam.size(1),cam.size(2)

        cam = F.relu(cam)
        cam=cam.mul(cls_label).cpu().numpy()

        norm_cam = cam / (np.max(cam, (1, 2), keepdims=True) + 1e-5)
        bg_score = [np.ones_like(norm_cam[0]) * bg_threshold]
        cam_predict = np.argmax(np.concatenate((bg_score, norm_cam)), 0)

        label = np.unique(cam_predict)
        label = label[1:]  # get the label except background
        # for each class

        bounding_box = {}
        bounding_scores = {}

        roi_index=[]
        label_list=[]

        for l in label:
            label_i = np.zeros((W,H))
            label_i[cam_predict == l] = 255
            label_i = np.uint8(label_i)
            contours, hier = cv2.findContours(label_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w < 4 or h < 4 or (h / w) > 4 or (w / h) > 4:  # filter too small bounding box
                    continue
                else:
                    xmin = x
                    ymin = y
                    xmax = x + w
                    ymax = y + h

                    bbox_region_cam = norm_cam[l-1, ymin:ymax, xmin:xmax]
                    bbox_score = np.average(bbox_region_cam)
                    if l not in bounding_box:
                        bounding_box[l] = list([[xmin, ymin, xmax, ymax]])
                        bounding_scores[l] = [bbox_score]
                    else:
                        bounding_box[l].append(list([xmin, ymin, xmax, ymax]))
                        bounding_scores[l].append(bbox_score)
            # NMS step
            if l in bounding_box:
                b = torch.from_numpy(np.array(bounding_box[l],np.double)).cuda()
                s = torch.from_numpy(np.array(bounding_scores[l],np.double)).cuda()
                bounding_box_index = torchvision.ops.nms(b,s,iou_threshhold)

                for i in bounding_box_index:
                    # generate new small dataset
                    xmin = bounding_box[l][i][0]
                    ymin = bounding_box[l][i][1]
                    xmax = bounding_box[l][i][2]
                    ymax = bounding_box[l][i][3]

                    roi_index.append(list([xmin, ymin, xmax, ymax]))
                    label_list.append(l)
                    if len(label_list)>3:
                        # print("2")
                        break

        # generating background object proposals
        bg_score = [np.ones_like(norm_cam[0]) * 0.05]
        cam_predict = np.argmax(np.concatenate((bg_score, norm_cam)), 0)


        # max_bg_area=2500
        # bg_box=None
        # label_i = np.zeros((W, H))
        # label_i[cam_predict == l] = 255
        # label_i = np.uint8(label_i)
        # contours, hier = cv2.findContours(label_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for c in contours:
        #     x, y, w, h = cv2.boundingRect(c)
        #     bg_area=w*h
        #     if bg_area<max_bg_area and w > 4 or h > 4:
        #         xmin = x
        #         ymin = y
        #         xmax = x + w
        #         ymax = y + h
        #
        #         bg_box=list([xmin, ymin, xmax, ymax])
        #
        # if bg_box:
        #     roi_index.append(bg_box)
        #     label_list.append(l)
        #
        # # NMS step
        # if len(label_list)==0:
        #     temp=0

        l = 0
        label_i = np.zeros((W, H))
        label_i[cam_predict == l] = 255
        label_i = np.uint8(label_i)
        contours, hier = cv2.findContours(label_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 9 or h < 9 or (h / w) > 4 or (w / h) > 4:  # filter too small bounding box
                continue
            else:
                xmin = x
                ymin = y
                xmax = x + w
                ymax = y + h

                bbox_region_cam = norm_cam[l - 1, ymin:ymax, xmin:xmax]
                bbox_score = np.average(bbox_region_cam)
                if l not in bounding_box:
                    bounding_box[l] = list([[xmin, ymin, xmax, ymax]])
                    bounding_scores[l] = [bbox_score]
                else:
                    bounding_box[l].append(list([xmin, ymin, xmax, ymax]))
                    bounding_scores[l].append(bbox_score)
        # NMS step
        if l in bounding_box:
            b = torch.from_numpy(np.array(bounding_box[l], np.double)).cuda()
            s = torch.from_numpy(np.array(bounding_scores[l], np.double)).cuda()
            bounding_box_index = torchvision.ops.nms(b, s, iou_threshhold)

            for i in bounding_box_index:
                # generate new small dataset
                xmin = bounding_box[l][i][0]
                ymin = bounding_box[l][i][1]
                xmax = bounding_box[l][i][2]
                ymax = bounding_box[l][i][3]

                roi_index.append(list([xmin, ymin, xmax, ymax]))
                label_list.append(100)
                if len(label_list) > 5:
                    # print("2")
                    break

        return roi_index, label_list

    # 算两个feature map的欧几里得距离; 在metric learning中用的
    def euclidean_dist(self,x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """

        m, n = x.size(0), y.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
        dist.addmm_(1, -2, x, y.t())
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    # 基于得到的bounding box和label, 变成feature vector后计算metric learning loss
    def patch_based_metric_loss(self, img, x_patch, label, patch_num=4, is_same_img=False, is_hard_negative=True):

        N_f, _, patch_w, patch_h = x_patch.size()
        cam_wo_dropout1 = self.fc8_(x_patch.detach())
        # 得到每一张图片每一类对应的激活区的patch_num块区域的特征激活向量 & 对应的标签 & 来自的图片的索引
        roi_cls_label = []
        roi_cls_feature_vector = []  # torch.Tensor([]).cuda()
        img_ids = []

        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 可对这一部分进行改进
        # begin

        
        for i in range(N_f):# for 每张图片
            # 确定boundingbox
            if is_same_img:
                roi_index, label_list = self.get_roi_index_patch_same_image(cam_wo_dropout1[i].detach(), label[i])
            else:
                roi_index, label_list = self.get_roi_index(cam_wo_dropout1[i].detach(), label[i])
            # if len(label_list) == 0:
            #     print(i)
            
            if(roi_index!=[]):
                # --------------------- BEGIN LBBA ----------------------------------- 
                # lbba输入的bbox对应的是原图上的坐标，因此对应到特征图上坐标的roi_index需要转变一下尺度
                roi_index = [[0]+roi for roi in roi_index]
                roi_index = torch.tensor(roi_index, dtype=torch.float)
                w_scale, h_scale = img.shape[2]/patch_w, img.shape[3]/patch_h
                if(w_scale!=h_scale):
                    raise ValueError("w_scale!=h_scale")
                # for colidx in range(1,5):
                #     if(colidx==1 or colidx ==3):
                #         roi_index[:,colidx]=roi_index[:,colidx]*w_scale
                #     else:
                #         roi_index[:,colidx]=roi_index[:,colidx]*h_scale   
                roi_index = roi_index * w_scale
                # 使用lbba预训练的teacher模型对初步的roi进行修正
                _, _, bbox_pred_t = self.teacher.module.inference_step(img[i], roi_index)
                bbox_pred_t = bbox_pred_t.detach()
                roi_index = torch.stack([temp[1:] for temp in roi_index])
                rectified_roi_index = bbox_transform_inv(roi_index, bbox_pred_t.cpu())
                clipped_roi_index = clip_boxes(rectified_roi_index, [img[i].shape[1], img[i].shape[2]])
                # 修正后的bbox对应的也是原图上的坐标，需要转回特征图上的坐标
                # lbba的输出是浮点数，但是我们的方法后面也要对proposal进行切块，要对坐标进行处理，所以这里直接保留浮点数
                # for colidx in range(0,4):
                #     if(colidx==1 or colidx ==3):
                #         clipped_roi_index[:,colidx]=clipped_roi_index[:,colidx]/w_scale
                #     else:
                #         clipped_roi_index[:,colidx]=clipped_roi_index[:,colidx]/h_scale   
                clipped_roi_index = clipped_roi_index / w_scale
                roi_index = clipped_roi_index.numpy().tolist()
                # --------------------- END LBBA -----------------------------------

            if len(label_list) > 0:   # 用cams激活区域的boundingbox去框在最后一组特征（256维）中的对应位置，然后切成patch_num份，每一份中每一维做一个全局平均池化
                roi_cls_pooled, roi_pool_mask = self.roi_pooling(x_patch[i], roi_index, patch_num)  # predict roi_cls_label

                # roi_cls_pooled = torch.squeeze(roi_cls_pooled)  # [batch_num*4096]

                #########2022,增加向量归一化
                # roi_cls_pooled=roi_cls_pooled / roi_cls_pooled.norm(dim=-1, keepdim= True)
                if len(roi_cls_pooled) > 0:
                    roi_cls_pooled = roi_cls_pooled.squeeze()
                    roi_cls_feature_vector.append(roi_cls_pooled)
                    temp = list(label_list) * patch_num      # [1,2,3]*3=[1, 2, 3, 1, 2, 3, 1, 2, 3]
                    temp_idx = range(len(temp))
                    roi_cls_label.extend([a for (a,b) in zip(temp, temp_idx) if roi_pool_mask[b]==1]) 
                    # img_ids.extend([i]*len(label_list)*patch_num)
                    img_ids.extend([i]*len([1 for temp in roi_pool_mask if temp==1]))
        
        # if(len(roi_cls_feature_vector)!=len(roi_cls_label) or len(roi_cls_label)!=len(img_ids)):
        #     raise ValueError("something wrong, [len(roi_cls_feature_vector)!=len(roi_cls_label) or len(roi_cls_label)!=len(img_ids)]")
        # end 
        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        patch_labels = torch.from_numpy(np.asarray(roi_cls_label)).cuda()
        img_ids = torch.from_numpy(np.asarray(img_ids)).cuda()

        if len(roi_cls_feature_vector) == 0:
            n = 0
        else:
            patch_embs = torch.cat(roi_cls_feature_vector, 0)

            n = patch_embs.size(0)  
            # =============== euclidean distance + triplet loss with hard sample mining ===============

            # try to use metric learning lib directly : the function below use cosine similarity
            # hard_pairs = miner(patch_embs, patch_labels)
            # self.loss_patch_cls = loss_func(patch_embs, patch_labels, hard_pairs)

            # ======
            distance = self.euclidean_dist(patch_embs, patch_embs)

            # For each anchor, find the hardest positive and negative
            mask = patch_labels.expand(n, n).eq(patch_labels.expand(n, n).t())
            img_mask = img_ids.expand(n, n).eq(img_ids.expand(n, n).t())

        dist_ap, dist_an = [], []

        # is_hard_negative = True

        # for
        for i in range(n):
            if patch_labels[i].item()==100:
                continue
            if is_same_img:
                # neg_idx=~mask[i] & img_mask[i]
                neg_idx=~mask[i]
            else:
                neg_idx = ~mask[i]
            an_i = distance[i][neg_idx]
            if an_i.size(0) == 0:
                continue
            if is_same_img:
                # pos_idx=mask[i] & img_mask[i] # 这个限制了学习图片之前的同类别相似性
                pos_idx=mask[i]
            else:
                pos_idx=mask[i]
            if is_hard_negative:
                dist_ap.append(distance[i][pos_idx].max().unsqueeze(0))
                dist_an.append(an_i.min().unsqueeze(0))
            else:
                # dist_ap.append(distance[i][pos_idx].topk(4).values[random.randint(0, 3)].unsqueeze(0))
                # dist_an.append(an_i.topk(4, largest=False).values[random.randint(0, 3)].unsqueeze(0))
                dist_ap.append(distance[i][pos_idx][random.randint(0, distance[i][pos_idx].shape[0]-1)].unsqueeze(0))
                dist_an.append(an_i[random.randint(0, an_i.shape[0]-1)].unsqueeze(0))    # 如果一个batch里面没有不同类别的图片会报错

        # triplet loss
        if len(dist_ap)>0:
            dist_ap = torch.cat(dist_ap, 0)
            dist_an = torch.cat(dist_an, 0)

            y = torch.ones_like(dist_an)

            if is_same_img:
                loss_patch_cls = self.ranking_loss_same_img(dist_an, dist_ap, y) / y.shape[0]   # 如果有bbox，正对肯定是有的，负对就不一定了
            else:
                loss_patch_cls = self.ranking_loss(dist_an, dist_ap, y) / y.shape[0]

        # # ================
            return loss_patch_cls
        else:
            return torch.tensor(0.0).cuda()

    # 消融用的
    def patch_based_metric_loss_cam(self, x_wo_dropout, label, patch_num=4, is_same_img=False):

        N_f, _, _, _ = x_wo_dropout.size()
        cam_wo_dropout1 = x_wo_dropout.detach()

        roi_cls_label = []
        roi_cls_feature_vector = []  # torch.Tensor([]).cuda()
        img_ids = []
        for i in range(N_f):# for 每张图片
            if is_same_img:
                # roi_index, label_list = self.get_roi_index_patch_same_image(cam_wo_dropout1[i].detach(), label[i])
                roi_index, label_list = self.get_roi_index_patch_same_image(cam_wo_dropout1[i].detach(), label[i])
            else:
                roi_index, label_list = self.get_roi_index(cam_wo_dropout1[i].detach(), label[i])

            if len(label_list) > 0:
                roi_cls_pooled = self.roi_pooling(x_wo_dropout[i], roi_index, patch_num).squeeze()  # predict roi_cls_label

                # roi_cls_pooled = torch.squeeze(roi_cls_pooled)  # [batch_num*4096]

                #########2022,增加向量归一化
                # roi_cls_pooled=roi_cls_pooled / roi_cls_pooled.norm(dim=-1, keepdim= True)

                roi_cls_feature_vector.append(roi_cls_pooled)

                roi_cls_label.extend(list(label_list) * patch_num)
                img_ids.extend([i]*len(label_list)*patch_num)

        patch_labels = torch.from_numpy(np.asarray(roi_cls_label)).cuda()
        img_ids = torch.from_numpy(np.asarray(img_ids)).cuda()

        patch_embs = torch.cat(roi_cls_feature_vector, 0)

        n = patch_embs.size(0)
        # =============== euclidean distance + triplet loss with hard sample mining ===============

        # try to use metric learning lib directly : the function below use cosine similarity
        # hard_pairs = miner(patch_embs, patch_labels)
        # self.loss_patch_cls = loss_func(patch_embs, patch_labels, hard_pairs)

        # ======
        distance = self.euclidean_dist(patch_embs, patch_embs)

        # For each anchor, find the hardest positive and negative
        mask = patch_labels.expand(n, n).eq(patch_labels.expand(n, n).t())
        img_mask = img_ids.expand(n, n).eq(img_ids.expand(n, n).t())

        dist_ap, dist_an = [], []

        is_hard_negative = True

        # for
        for i in range(n):
            if patch_labels[i].item()==100:
                continue
            if is_same_img:
                neg_idx=~mask[i] & img_mask[i]
            else:
                neg_idx = ~mask[i]
            an_i = distance[i][neg_idx]
            if an_i.size(0) == 0:
                if is_same_img:
                    # neg_idx = ~mask[i]
                    # an_i = distance[i][neg_idx]
                    # print("not here right? \n")
                    continue
                else:
                    continue

            if is_same_img:
                # pos_idx=mask[i] & img_mask[i]
                pos_idx=mask[i]
            else:
                pos_idx=mask[i]
            if is_hard_negative:
                dist_ap.append(distance[i][pos_idx].max().unsqueeze(0))
                dist_an.append(an_i.min().unsqueeze(0))
            else:
                dist_ap.append(distance[i][pos_idx].topk(4).values[random.randint(0, 3)].unsqueeze(0))
                dist_an.append(an_i.topk(4, largest=False).values[random.randint(0, 3)].unsqueeze(0))

        # triplet loss
        if len(dist_ap)>0:
            dist_ap = torch.cat(dist_ap, 0)
            dist_an = torch.cat(dist_an, 0)
        # print(dist_ap[:10])
        # print(dist_an[:10])
            y = torch.ones_like(dist_an)

            if is_same_img:
                loss_patch_cls = self.ranking_loss_same_img(dist_an, dist_ap, y) / y.shape[0]
            else:
                loss_patch_cls = self.ranking_loss(dist_an, dist_ap, y) / y.shape[0]

        # # ================
            return loss_patch_cls
        else:
            return torch.tensor(0.0).cuda()

    # 
    def forward(self, x, bounding_box=None, label=None, param=None, is_patch_metric=True, patch_num=4,is_sse=False):
        N, C, H, W = x.size()
        img = x
        # keep the feature map without dropout
        x = super().forward(x)
        x = self.dropout7(x)
        x = self.downsample(x)
        x_patch = F.relu(x)         # 最后一组特征
        x2 = self.fc8_(x_patch)     # CAMs

        cam = F.interpolate(x2, (H, W), mode='bilinear')    # resize到跟目前的图片一样的大小

        if label is not None:
            # multi-label soft margin loss:
            predicts = F.adaptive_avg_pool2d(cam, (1, 1))  # GAP的作用，得到各类置信度
            loss_cls = F.multilabel_soft_margin_loss(predicts, label)


            result = [loss_cls]

            if is_patch_metric:
                patch_metric_loss=self.patch_based_metric_loss(img, x_patch, label, is_same_img=False, is_hard_negative=True)
                # patch_metric_loss=self.patch_based_metric_loss_cam(x2, label, is_same_img=True)
                # patch_metric_loss_9=self.patch_based_metric_loss(x_wo_dropout, label,patch_num=9,)
                # patch_metric_loss_same_img=self.patch_based_metric_loss(x_wo_dropout, label, is_same_img=True)
                # patch_loss= (patch_metric_loss+patch_metric_loss_same_img)/2
                # patch_metric_loss= patch_metric_loss_same_img
                # patch_metric_loss= patch_metric_loss_same_img
                return [loss_cls, patch_metric_loss]

        else:
            result = cam

        return result

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
