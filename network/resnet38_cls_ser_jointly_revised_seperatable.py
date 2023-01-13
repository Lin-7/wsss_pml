import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tool.RoiPooling_Jointly import RoiPooling, RoiPoolingRandom, RoiPoolingContrastive
import network.resnet38d

import cv2
import torchvision
import random
# import math
# from sklearn.cluster import KMeans
# from sklearn.decomposition import  PCA
# from collections import Counter
import os
import PIL.Image
import matplotlib.pyplot as plt
from voc12.data import get_img_path
from tool.imutils import reNormalize

# from tool import pyutils
# seed = pyutils.seed_everything()

categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

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


    def patch_based_metric_init(self,):

        # 目标1：降低特征的维度，便于计算
        # 目标2：多了一个非线性层，给模型多一些参数来学习我们施加的patch-based metric learning
        self.downsample = nn.Conv2d(4096, 256, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.downsample.weight)
        self.from_scratch_layers.append(self.downsample)

        # 平均池化，得到CAM的bounding box后，要对这一个区域中的特征做一个平均池化
        # 1 先align box的位置
        # 2 对box内的feature map做一个平均池化
        
        self.ranking_loss = nn.MarginRankingLoss(margin=28.0)
        self.ranking_loss_same_img = nn.MarginRankingLoss(margin=28.0)
    
    def init_roi_pooling_method(self, args):
        if args.patch_gen == "4patch":
            self.roi_pooling = RoiPooling(mode="th")
        elif args.patch_gen == "randompatch":
            self.roi_pooling = RoiPoolingRandom(mode="th")
        elif args.patch_gen == "contrastivepatch":
            self.roi_pooling = RoiPoolingContrastive(mode="th", args=args)
        else:
            raise ValueError(f"patch_gen must be 4patch, randompatch or contrastivepatch, but now it is {args.patch_gen}")

    # 输入CAM图片和对应的类别
    # 返回bounding box的坐标和对应的类别
    def get_roi_index(self, cam, cls_label, padding=0):
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
        W,H = cam.size(1),cam.size(2)    # 这里感觉应该是H，W，但W和H是一样的所以没关系，后面的所有都把x与w对应

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

                    bbox_region_cam = norm_cam[l-1, ymin:ymax, xmin:xmax]  # 因为cv2的xy分别对应的是横轴和纵轴，而np二维数组或者torch的二维tensor的xy分别对应纵轴和横轴，两者是相反的
                    bbox_score = np.average(bbox_region_cam)
                    if l not in bounding_box:
                        bounding_box[l] = list([[xmin, ymin, xmax, ymax]])    # 这个xy是cv2对应的
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
                    
                    if padding != 0:
                        w, h = xmax-xmin, ymax-ymin
                        xmin = max(0, xmin-int(w*padding))
                        ymin = max(0, ymin-int(h*padding))
                        xmax = min(W, xmax+int(w*padding))
                        ymax = min(H, ymax+int(h*padding))

                    roi_index.append(list([xmin, ymin, xmax, ymax]))
                    label_list.append(l)
                    if len(label_list)>=3:    # 每张图片，每个label最多返回三个轮廓
                        # print("2")
                        # return roi_index, label_list
                        break
        return roi_index, label_list, cam_predict  # 这里的坐标先横轴再纵轴

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
        # dist.addmm_(1, -2, x, y.t())
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    # 基于得到的bounding box和label, 变成feature vector后计算metric learning loss
    # patch_nums=[1, 4, 9, 16, 25]
    def patch_based_metric_loss(self, img, x_patch, label, bboxes, bbxs_cls, bbxs_img, patch_nums=[4], 
                    is_same_img=False, is_hard_negative=True, epoch_iter="null", img_names=[], args=""):

        N_f, _, patch_w, patch_h = x_patch.size()
        assert patch_w==patch_h, "特征图长宽不等，下面特征图坐标与原图坐标的对齐操作可能会出错"
        cam_wo_dropout1 = self.fc8_(x_patch.detach())
        # 得到每一张图片每一类对应的激活区的patch_num块区域的特征激活向量 & 对应的标签 & 来自的图片的索引
        roi_cls_label = []
        roi_cls_feature_vector = []  # torch.Tensor([]).cuda()
        fg_roi_cls_feature_vector = []
        nfg_roi_cls_feature_vector = []
        img_ids = []
        scores = []
        patch_locs = []
        triplet_info = []
        patch_embs = []

        # # 将图片坐标转换为特征图上的坐标
        # f2i_scale_w = patch_w/img.size(2)
        # f2i_scale_h = patch_h/img.size(3)
        # gt_bbx_mask = []
        # for b_idx in range(len(bboxes)):
        #     bboxes[b_idx][0] *= f2i_scale_w
        #     bboxes[b_idx][1] *= f2i_scale_h
        #     bboxes[b_idx][2] *= f2i_scale_w
        #     bboxes[b_idx][3] *= f2i_scale_h
        #     bw = bboxes[b_idx][2]-bboxes[b_idx][0]
        #     bh = bboxes[b_idx][3]-bboxes[b_idx][1]
        #     if bw < 9 or bh < 9 or (bh / bw) > 4 or (bw / bh) > 4:
        #         gt_bbx_mask.append(0)
        #     else:
        #         gt_bbx_mask.append(1)
        # gt_bbx_mask = np.array(gt_bbx_mask)
        # bboxes = bboxes[gt_bbx_mask==1]
        # bbxs_cls = bbxs_cls[gt_bbx_mask==1]
        # bbxs_img = bbxs_img[gt_bbx_mask==1]
        proposal_num = 0 
        
        for i in range(N_f):# for 每张图片
            # 确定boundingbox
            if is_same_img:
                roi_index, label_list = self.get_roi_index_patch_same_image(cam_wo_dropout1[i].detach(), label[i])
            else:
                roi_index, label_list, mask_predict = self.get_roi_index(cam_wo_dropout1[i].detach(), label[i], args.proposal_padding)   # 检查最大值是否与特征图长度一致--是的
                # label_list中label范围1-20
                # bbox_idxs = bbxs_img==i
                # roi_index = bboxes[bbox_idxs]
                # label_list = bbxs_cls[bbox_idxs]

            if len(label_list) > 0:   # 用cams激活区域的boundingbox去框在最后一组特征（256维）中的对应位置，然后切成patch_num份，每一份中每一维做一个全局平均池化
                proposal_num += len(label_list)
                roi_cls_pooled, roi_label_list, score_list, p_locs \
                    = self.roi_pooling(feature_map=x_patch[i], roi_batch=roi_index, label_list=label_list, \
                        patch_nums=patch_nums, cam_predict=mask_predict, cam=cam_wo_dropout1[i].detach(), cls_label=label[i])  # predict roi_cls_label

                if len(roi_cls_pooled) > 0:

                    scores.extend(score_list)
                    roi_cls_feature_vector.append(roi_cls_pooled)
                    # fg_roi_cls_feature_vector.append(fg_roi_cls_pooled)
                    # nfg_roi_cls_feature_vector.append(nfg_roi_cls_pooled)
                    roi_cls_label.extend(roi_label_list)
                    img_ids.extend([i]*len(roi_label_list))
                    patch_locs.extend(p_locs)
        
        patch_labels = torch.from_numpy(np.asarray(roi_cls_label)).cuda()
        img_ids = torch.from_numpy(np.asarray(img_ids)).cuda()

        # 将特征图上的坐标与原图坐标对齐
        scale = img.size(2)/patch_w
        for i in range(len(patch_locs)):
            patch_locs[i] = [int(patch_locs[i][j]*scale) for j in range(4)]

        if len(roi_cls_feature_vector) == 0:
            n = 0
            # 当某一轮没有patch的时候，也要初始化返回变量，同时为了dataparallel多gpu能正常运作，需要这些变量的维度和有patch的时候一样（不然多gpu的返回结果无法正确拼接），同时变量都要在gpu上
            patch_embs =  torch.zeros(256).unsqueeze(0).cuda()
            patch_labels = torch.tensor([-1]).cuda()
            patch_mask = torch.tensor([-1]).cuda()
        else:
            patch_embs = torch.cat(roi_cls_feature_vector, 0)
            # fg_patch_embs = torch.cat(fg_roi_cls_feature_vector, 0)
            # nfg_patch_embs = torch.cat(nfg_roi_cls_feature_vector, 0)

            # 以batch中的所有图片为单位挑patch
            # 按分数(前景占比,置信度）选择指定比例的patches
            sorted_idxes = np.argsort(scores)[::-1]   # 按分数(前景占比,置信度）降序排序
            As = proposal_num * 4 * args.patch_select_ratio
            Ag = len(patch_labels)
            if args.patch_select_part == "front":
                indexs = np.sort(sorted_idxes[:int(As)].copy())
            elif args.patch_select_part == "mid":
                bgn = (Ag-As)/2/Ag
                end = 1-bgn
                if(end<=bgn):
                    raise ValueError("end must greater than bgn!!!!")
                bgn_idx = int((Ag - As)/2)
                indexs = np.sort(sorted_idxes[bgn_idx:bgn_idx+int(As)].copy())
            elif args.patch_select_part == "back":
                indexs = np.sort(sorted_idxes[-int(As):].copy())
            elif args.patch_select_part == "random":
                indexs = np.random.choice(range(len(scores)), size=int(As), replace=False, p=None)
            else:
                raise ValueError(f"'patch_select_part' must be front, mid, back or random. But now 'patch_select_part' is {args.patch_select_part}!!!!")
            
            # 选择所有的patches
            # indexs = range(len(scores))
            patch_mask = np.zeros(len(patch_labels))
            patch_mask[indexs] = 1
            patch_mask = torch.tensor(patch_mask).cuda()

            patch_embs_select = patch_embs[indexs]
            # fg_patch_embs_select = fg_patch_embs[indexs]
            # nfg_patch_embs_select = nfg_patch_embs[indexs]
            patch_labels_select = patch_labels[indexs]
            img_ids_selected = img_ids[indexs]

            patch_embs = patch_embs.detach()

            n = patch_embs_select.size(0)  
            # =============== euclidean distance + triplet loss with hard sample mining ===============

            # try to use metric learning lib directly : the function below use cosine similarity
            # hard_pairs = miner(patch_embs, patch_labels)
            # self.loss_patch_cls = loss_func(patch_embs, patch_labels, hard_pairs)

            # ======
            # patch相似度
            distance = self.euclidean_dist(patch_embs_select, patch_embs_select)
            # # 目标前景相似度
            # fg_distance = self.euclidean_dist(fg_patch_embs_select, fg_patch_embs_select)
            # # 非目标前景区域的相似度
            # nfg_distance = self.euclidean_dist(nfg_patch_embs_select, nfg_patch_embs_select)

            # For each anchor, find the hardest positive and negative
            mask = patch_labels_select.expand(n, n).eq(patch_labels_select.expand(n, n).t())
            img_mask = img_ids_selected.expand(n, n).eq(img_ids_selected.expand(n, n).t())

        dist_ap, dist_an = [], []

        for i in range(n):
            if patch_labels_select[i].item()==100:
                continue
            if is_same_img:
                # neg_idx=~mask[i] & img_mask[i]
                neg_mask=~mask[i]
                neg_idx=torch.tensor(range(len(neg_mask)))[neg_mask==1]
            else:
                neg_mask=~mask[i]
                neg_idx=torch.tensor(range(len(neg_mask)))[neg_mask==1]
            an_i = distance[i][neg_mask]
            # an_i = fg_distance[i][neg_idx]
            if an_i.size(0) == 0:
                continue
            if is_same_img:
                # pos_idx=mask[i] & img_mask[i] # 这个限制了学习图片之前的同类别相似性
                pos_mask=mask[i]
                pos_idx=torch.tensor(range(len(pos_mask)))[pos_mask==1]
            else:
                pos_mask=mask[i]
                pos_idx=torch.tensor(range(len(pos_mask)))[pos_mask==1]
            if is_hard_negative:
                dist_ap.append(distance[i][pos_mask].max().unsqueeze(0))
                # dist_ap.append(fg_distance[i][pos_mask].max().unsqueeze(0))
                dist_an.append(an_i.min().unsqueeze(0))
                p_idx = distance[i][pos_mask].argmax()
                n_idx = an_i.argmin()
            else:
                p_idx = random.randint(0, distance[i][pos_mask].shape[0]-1)
                n_idx = random.randint(0, an_i.shape[0]-1)
                # dist_ap.append(distance[i][pos_mask].topk(4).values[random.randint(0, 3)].unsqueeze(0))
                # dist_an.append(an_i.topk(4, largest=False).values[random.randint(0, 3)].unsqueeze(0))
                # dist_ap.append(fg_distance[i][pos_mask][random.randint(0, fg_distance[i][pos_mask].shape[0]-1)].unsqueeze(0))
                dist_ap.append(distance[i][pos_mask][p_idx].unsqueeze(0))
                dist_an.append(an_i[n_idx].unsqueeze(0))    # 如果一个batch里面没有不同类别的图片会报错
            
            # 记录的是三元组中anchor positive negative各自对应的在当前batch的所有patch下的下标
            triplet_info.append((indexs[i], indexs[pos_idx[p_idx]], indexs[neg_idx[n_idx]]))

        '''
        # TODO: 还没完成的一些选triplet的尝试
        # for i in range(n):
        #     # if patch_labels[i].item()==100:
        #     #     continue
        #     # TODO：现在是直接选择前景最像的positive和背景最像的negative 还有些问题
        #     # TODO：1 选positive的时候可以先聚类，把同一聚类结果的拉近或者拉近里面最远的
            
        #     kmeans = KMeans(n_clusters=2, random_state=1234)
        #     # pca = PCA(n_components=10, random_state=1234)
        #     pca = PCA(n_components=0.9, svd_solver='full', random_state=1234)
        #     # pca = PCA(n_components=10, svd_solver='randomized', random_state=1234)
        #     # TODO：这里an感觉直接用nfg_distance选最近的非同前景patch就好了--no hard sampling的选择范围就小了
        #     # an
        #     neg_idx = ~mask[i]
        #     # # 用nfg特征聚成两类，确定与当前patch处于同一聚类的patch
        #     # i_n_nfg_patch_embs = nfg_patch_embs[neg_idx].clone().cpu().detach()
        #     # if i_n_nfg_patch_embs.size(0)<1:
        #     #     continue
        #     # i_n_nfg_patch_embs = torch.cat([i_n_nfg_patch_embs, nfg_patch_embs[i].clone().cpu().detach().unsqueeze(0)],0)
        #     # pca_i_n = pca.fit_transform(i_n_nfg_patch_embs.numpy())
        #     # clusterresult_i_n = kmeans.fit(pca_i_n).labels_
        #     # if clusterresult_i_n[-1] == 0:
        #     #     clusterresult_i_n = ~(clusterresult_i_n.astype(bool))[:-1]
        #     # else:
        #     #     clusterresult_i_n = clusterresult_i_n.astype(bool)[:-1]
        #     # n_count = np.sum(clusterresult_i_n)
            
        #     # ap
        #     pos_idx = mask[i]
        #     pos_idx[i] = 0
        #     # 用fg特征聚成两类，确定与当前patch处于同一聚类的patch
        #     i_p_fg_patch_embs = (patch_embs[pos_idx]+0).cpu().detach()
        #     if i_p_fg_patch_embs.size(0)<1:
        #         continue
        #     i_p_fg_patch_embs1 = torch.cat([i_p_fg_patch_embs, (patch_embs[i]+0).cpu().detach().unsqueeze(0)],0)
        #     pca_i_p = pca.fit_transform(i_p_fg_patch_embs1.numpy())
        #     clusterresult_i_p = kmeans.fit_predict(pca_i_p)
        #     if clusterresult_i_p[-1] == 0:
        #         clusterresult_i_p1 = ~(clusterresult_i_p.astype(bool))[:-1]
        #     else:
        #         clusterresult_i_p1 = clusterresult_i_p.astype(bool)[:-1]
        #     p_count = np.sum(clusterresult_i_p1)


        #     # 没有则跳过,有则通过下标确定距离并加入到dist_ap和dist_an中
        #     # if p_count>0:
        #     an_i_dist = distance[i][neg_idx]
        #     an_i_nfg_dist = nfg_distance[i][neg_idx]
        #     # an_i_dist1 = an_i_dist[clusterresult_i_n]
        #     # 在背景区域相似的patch中，选前景最近的
        #     dist_an.append(an_i_dist[torch.argmin(an_i_nfg_dist)].unsqueeze(0))
            
        #     # ap_i_dist = distance[i][pos_idx]
        #     # ap_i_dist1 = ap_i_dist[clusterresult_i_p]
        #     # dist_ap.append(ap_i_dist1[torch.argmax(ap_i_dist1)].unsqueeze(0))

        #     # neg_idx = ~mask[i]
        #     # an_i_dist = distance[i][neg_idx]
        #     # dist_an.append(an_i_dist[torch.argmin(an_i_dist)].unsqueeze(0))
        #     pos_idx = mask[i]
        #     ap_i_dist = distance[i][pos_idx]
        #     dist_ap.append(ap_i_dist[torch.argmax(ap_i_dist)].unsqueeze(0))
            
        #     # tried:ap只挑前景最相似（特征欧式距离最近）的一个， an只挑背景最相似（同）的一个；
        #     # ap挑前景最相似的所有（通过二聚类结果选择同一子聚类下的所有patch），an挑背景最相似的所有（同）(no,每个patch只能挑一正一负）；
        #     # ap挑前景最相似里面最远的（hard sampling），an挑背景最相似里面前景最近的（hard sampling）
        #     # dist_an.append(an_i[torch.argmin(an_i_nfg)].unsqueeze(0))
        #     # dist_ap.append(ap_i[torch.argmin(ap_i)].unsqueeze(0))        
        '''

        # triplet loss
        if len(dist_ap)>0:
            # 可视化当前batch的patches
            if epoch_iter!="null":
                self.visualize_batch_patches(img.detach().cpu(), img_names, patch_locs, img_ids.cpu(), patch_labels.cpu(), patch_mask.cpu(), scores, triplet_info, args, epoch_iter)   # 把cuda上的东西都放到cpu上，在计算图中的都detach下来

            dist_ap = torch.cat(dist_ap, 0)
            dist_an = torch.cat(dist_an, 0)

            y = torch.ones_like(dist_an)

            if is_same_img:
                loss_patch_cls = self.ranking_loss_same_img(dist_an, dist_ap, y) / y.shape[0]   # 如果有bbox，正对肯定是有的，负对就不一定了
            else:
                loss_patch_cls = self.ranking_loss(dist_an, dist_ap, y) / y.shape[0]

            return loss_patch_cls, patch_embs, patch_labels, patch_mask
        else:
            return torch.tensor(0.0).cuda(), patch_embs, patch_labels, patch_mask

    def forward(self, x, bounding_box=None, label=None, img_names=[], param=None, is_patch_metric=True, patch_nums=[4],is_sse=False,featuremap=False, \
        patches=False, epoch_iter="null", args=""):

        N, C, W, H = x.size()
        img = x
        # keep the feature map without dropout
        x = super().forward(x)
        x = self.dropout7(x)
        x = self.downsample(x)
        x_patch = F.relu(x)         # 最后一组特征
        x2 = self.fc8_(x_patch)     # CAMs

        if args.interpolate_mode == "bilinear":
            cam = F.interpolate(x2, (W, H), mode=args.interpolate_mode, align_corners=False)    # resize到跟目前的图片一样的大小
        else:
            cam = F.interpolate(x2, (W, H), mode=args.interpolate_mode)    # resize到跟目前的图片一样的大小

        if featuremap:
            return x_patch
        
        if patches:
            patches_list, label_list = [], []
            x2 = x2.detach()
            for i in range(x2.size(0)):
                proposals, proposal_labels, mask_predict = self.get_roi_index(x2[i], label[i])

                if len(proposal_labels)>0:
                    roi_cls_pooled, roi_label_list = self.roi_pooling(x_patch[i], proposals, proposal_labels, patch_nums, mask_predict)
                    
                    if len(roi_cls_pooled)>0:
                        patches_list.append(roi_cls_pooled.detach().cpu().numpy())
                        label_list.append(roi_label_list)
            return patches_list, label_list

        if label is not None:
            # multi-label soft margin loss:
            predicts = F.adaptive_avg_pool2d(cam, (1, 1))  # GAP的作用，得到各类置信度
            loss_cls = F.multilabel_soft_margin_loss(predicts, label)

            result = [loss_cls]

            if is_patch_metric:

                w, h, raw_W, raw_H, _ = param  # 检查bbx的数量是否对应x的, wh不同的位置写得不一样，但注意前面那个一直到在图片中裁剪图片时都是对应前面的
                bbxs_cls = []
                bboxes = []
                bbxs_img  = []
                device = torch.cuda.current_device()
                for i in range((device)*N, (device+1)*N):
                    bbox = bounding_box[i]
                    for key in bbox.keys():
                        for j in range(len(bbox[key])):
                            bbox[key][j][0] *= float(w)/raw_W[i%N].item()  # xmin
                            bbox[key][j][1] *= float(h)/raw_H[i%N].item()   # ymin
                            bbox[key][j][2] *= float(w)/raw_W[i%N].item()   # xmax
                            bbox[key][j][3] *= float(h)/raw_H[i%N].item()  # ymax
                            bboxes.append(bbox[key][j])
                            bbxs_cls.append(key)
                            bbxs_img.append(i-device*N)

                patch_metric_loss, patch_embs, patch_labels, patch_mask\
                    =self.patch_based_metric_loss(img, x_patch, label, np.array(bboxes), np.array(bbxs_cls), np.array(bbxs_img), is_same_img=False, is_hard_negative=True, epoch_iter=epoch_iter, img_names=img_names, args=args)
                # patch_metric_loss=self.patch_based_metric_loss_cam(x2, label, is_same_img=True)
                # patch_metric_loss_9=self.patch_based_metric_loss(x_wo_dropout, label,patch_num=9,)
                # patch_metric_loss_same_img=self.patch_based_metric_loss(x_wo_dropout, label, is_same_img=True)
                # patch_loss= (patch_metric_loss+patch_metric_loss_same_img)/2
                # patch_metric_loss= patch_metric_loss_same_img
                # patch_metric_loss= patch_metric_loss_same_img
                return [loss_cls, patch_metric_loss, patch_embs, patch_labels, patch_mask]

        else:
            result = cam

        return result

    def visualize_batch_patches(self, imgs, img_names, patch_locs, patch_img_labels, patch_labels, patch_mask, patch_scores, triples, args, epoch_iter):

        device = torch.cuda.current_device()
        if device==1:   # 两张卡上的patches都可视化的话，第一张卡的triplets图画不出来，所以只画第一张卡的
            return
        img_names = img_names[imgs.shape[0]*device:imgs.shape[0]*(device+1)]
        imgs = imgs.numpy().transpose((0,2,3,1))
        imgs = reNormalize(imgs)           # 该函数接受hwc的输入
        img_list = [PIL.Image.fromarray(imgs[i]) for i in range(imgs.shape[0])]    # 该函数接受hwc的输入

        cur_save_dir = args.visualize_patch_dir + '/' + epoch_iter+f"_{torch.cuda.current_device()}"
        if not os.path.exists(cur_save_dir):
            os.mkdir(cur_save_dir)

        unselected_dir = cur_save_dir + '/unselected'
        if not os.path.exists(unselected_dir):
            os.mkdir(unselected_dir)

        selected_dir = cur_save_dir + '/selected'
        if not os.path.exists(selected_dir):
            os.mkdir(selected_dir)
        
        # 原图
        for i in range(len(imgs)):
            cur_img = PIL.Image.open(get_img_path(img_names[i], args.voc12_root)).convert("RGB")
            cur_img.save(unselected_dir+'/img{}.png'.format(img_names[i]))

        unselected_idxs = np.array(range(len(patch_mask)))[patch_mask == 0]
        for i in range(len(unselected_idxs)):
            origin_idx = unselected_idxs[i]
            cur_img = img_list[patch_img_labels[origin_idx]]
            patch = cur_img.crop(patch_locs[origin_idx])   # The crop rectangle, as a (left, upper, right, lower)-tuple.
            size = patch.size
            if size[0]*size[1] == 0:
                continue
            patch = patch.resize((300, int(300/size[0]*size[1])))
            patch.save(unselected_dir+'/img{}_{}_score{:.3f}.png'.format(img_names[patch_img_labels[origin_idx]], categories[patch_labels[origin_idx]-1], patch_scores[origin_idx]))

        selected_idxs = np.array(range(len(patch_mask)))[patch_mask == 1]
        for i in range(len(selected_idxs)):
            origin_idx = selected_idxs[i]
            cur_img = img_list[patch_img_labels[origin_idx]]
            patch = cur_img.crop(patch_locs[origin_idx])
            size = patch.size
            if size[0]*size[1] == 0:
                continue
            patch = patch.resize((300, int(300/size[0]*size[1])))
            patch.save(selected_dir+'/img{}_{}_score{:.3f}.png'.format(img_names[patch_img_labels[origin_idx]], categories[patch_labels[origin_idx]-1], patch_scores[origin_idx]))

        # 将triplet中的所有三元组可视化到一张图上（指定每行三张以及每张图的大小），所有triplet的anchor就是所有选择下来的patches
        lines = len(triples)
        pic_per_line = 3
        plt.figure(figsize=(12,90))

        for i in range(lines):
            for j in range(pic_per_line):
                plt.subplot(lines, pic_per_line, i*pic_per_line+j+1)

                origin_idx = triples[i][j]
                cur_img = img_list[patch_img_labels[origin_idx]]
                patch = cur_img.crop(patch_locs[origin_idx])
                plt.imshow(np.array(patch))

                plt.xticks([])
                plt.yticks([])
                plt.xlabel('img{}_{}_score{:.3f}'.format(img_names[patch_img_labels[origin_idx]], categories[patch_labels[origin_idx]-1], patch_scores[origin_idx]))

        plt.savefig(os.path.join(cur_save_dir, 'triplets.jpg'), bbox_inches='tight')

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
