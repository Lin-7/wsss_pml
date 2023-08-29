from torch.nn.modules.module import Module
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import time
import random
import math
from tool.ContrastiveCrop import ContrastiveCrop

# from tool import pyutils
# seed = pyutils.seed_everything()



class RoiPooling(Module):
    def __init__(self, mode='tf', pool_size=(1, 1), cls_layer=None):
        """
        tf: (height, width, channels)
        th: (channels, height, width)
        :param mode:
        :param pool_size:
        """
        super(RoiPooling, self).__init__()

        self.mode = mode
        self.pool_size = pool_size
        self.cls_layer = cls_layer

    def pool_region_ori(self, region):

        """
        the pooling of a region
        :param region: the region of interest fetched from feature map
        :return: roipool with size of (1, height, width, channel) if mode is tf otherwise (1, channels, height, width)
        """

        pool_height, pool_width = self.pool_size
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode== 'th':
            region_channels, region_height, region_width = region.shape
            pool = torch.from_numpy(np.zeros((region_channels, pool_height, pool_width)))

        h_step = region_height / pool_height
        w_step = region_width / pool_width
        for i in range(pool_height):
            for j in range(pool_width):

                xmin = j * w_step
                xmax = (j+1) * w_step
                ymin = i * h_step
                ymax = (i+1) * h_step

                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                # if xmin==xmax or ymin==ymax:
                #     continue
                if self.mode=='tf':
                    # pool[i, j, :] = np.max(region[ymin:ymax, xmin:xmax, :], axis=(0,1))
                    region_var = region[ymin:ymax, xmin:xmax, :]
                    region_cam = self.cls_layer(region_var)
                    pool[i,j,:] = F.adaptive_avg_pool2d(region_cam, (1,1))
                    # pool[i, j, :] = F.adaptive_max_pool2d(region[ymin:ymax, xmin:xmax, :], (1, 1))
                elif self.mode=='th':
                    region_var = region[:, ymin:ymax, xmin:xmax]
                    region_cam = self.cls_layer(region_var)
                    roi_pooled = F.adaptive_avg_pool2d(region_cam, (1, 1))

        return roi_pooled

    def pool_region(self, region):

        """
        the pooling of a region
        :param region: the region of interest fetched from feature map
        :return: roipool with size of (1, height, width, channel) if mode is tf otherwise (1, channels, height, width)
        """

        pool_height, pool_width = self.pool_size
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode== 'th':
            region_channels, region_height, region_width = region.shape
            pool = torch.from_numpy(np.zeros((region_channels, pool_height, pool_width)))

        if self.mode=='th':
            roi_pooled = F.adaptive_avg_pool2d(region, (1 ,1))

        return roi_pooled

    def get_region(self, feature_map, roi_dimensions):
        """
        fetching the roi from feature map by the dimension of the roi
        :param feature_map: the feature map with size of (1, height, width, channels)
        :param roi_dimensions: a region of interest dimensions
        :return:
        """
        xmin, ymin, xmax, ymax = roi_dimensions
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        if self.mode=='tf':
            r = np.squeeze(feature_map)[ymin:ymax, xmin:xmax, :]
        elif self.mode=='th':
            r = feature_map[:, ymin:ymax, xmin:xmax]
        return r
    
    def get_mask_region(self, mask, roi_dimensions):
        xmin, ymin, xmax, ymax = roi_dimensions
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        return mask[ymin:ymax, xmin:xmax]

    def forward(self,feature_map, roi_batch, label_list, patch_nums, norm_cam_bg, cls_label, area_ratios=[]):
        """
        getting pools from the roi batch
        :param feature_map:
        :param roi_batch: region of interest batch (usually is 256 for faster rcnn)
        :return:
        """
        # start=time.time()
        pool = []
        fg_pool = []
        nfg_pool = []
        pool_label_list = []
        W, H = feature_map.size(1), feature_map.size(2)
        i = 0
        fg_score = []
        confidence_score = []
        patch_locs = []

        cam_predict = np.argmax(norm_cam_bg, 0)
        
        for patch_num in patch_nums:

            patch_num_per_edge = int(math.sqrt(patch_num))

            for i in range(patch_num):

                for region_dim, region_label in zip(roi_batch, label_list):
                    map = feature_map
                    xmin,ymin,xmax,ymax=region_dim

                    # region_dim_i=[round(xmin+ (i % patch_num_per_edge) * (xmax - xmin) / patch_num_per_edge),
                    #             round(ymin + int(i / patch_num_per_edge) * (ymax - ymin) / patch_num_per_edge),
                    #             round(xmin+ (i % patch_num_per_edge) * (xmax - xmin) / patch_num_per_edge+(xmax - xmin) / patch_num_per_edge),
                    #             round(ymin + int(i / patch_num_per_edge) * (ymax - ymin) / patch_num_per_edge+(ymax - ymin) / patch_num_per_edge)
                    #             ]
                    region_dim_i=[int(xmin+ (i % patch_num_per_edge) * (xmax - xmin) / patch_num_per_edge),
                                int(ymin + int(i / patch_num_per_edge) * (ymax - ymin) / patch_num_per_edge),
                                math.ceil(xmin+ (i % patch_num_per_edge) * (xmax - xmin) / patch_num_per_edge+(xmax - xmin) / patch_num_per_edge),
                                math.ceil(ymin + int(i / patch_num_per_edge) * (ymax - ymin) / patch_num_per_edge+(ymax - ymin) / patch_num_per_edge)
                                ]
                    # print(region_dim_i)

                    # # 剔除过小的区域
                    # area = (region_dim_i[2]-region_dim_i[0])*(region_dim_i[3]-region_dim_i[1])
                    # # tried:10,25
                    # if area<10:                      # TODO: 需要调整的超参
                    #     continue
                    if region_dim_i[2]-region_dim_i[0]<=0 or region_dim_i[3]-region_dim_i[1]<=0:
                        print(region_dim_i[0])
                        print(region_dim_i[1])
                        print(region_dim_i[2])
                        print(region_dim_i[3])
                        continue

                    # 计算mask
                    label_i_mask = np.zeros((W,H))
                    label_i_mask[cam_predict == region_label] = 1
                    
                    # 计算前景占比
                    region_mask = self.get_mask_region(label_i_mask, region_dim_i)
                    target_percentage = np.sum(region_mask == 1) / np.size(region_mask)
                    fg_score.append(target_percentage)

                    # 剔除目标区域过小的区域
                    # if target_percentage < 0.5:      # TODO：需要调整的超参
                    # # if target_percentage == 0:      # TODO：需要调整的超参
                    #     # continue
                    #     filter_mask.append(0)
                    # else:
                    #     filter_mask.append(1)

                    # 计算patch的置信度分数
                    c_score = []
                    region_cam_predict = self.get_mask_region(cam_predict, region_dim_i)
                    label = np.unique(region_cam_predict)  # 0:bg
                    for l in label:
                        target_cam_region = self.get_mask_region(norm_cam_bg[l], region_dim_i)
                        c_score.append(target_cam_region[region_cam_predict == l].mean())
                    confidence_score.append(np.array(c_score).mean())

                    # 基于整个patch计算特征表示
                    region = self.get_region(map, region_dim_i)
                    p = self.pool_region_ori(region).squeeze().unsqueeze(0)   # [1,channel,1,1]
                    pool.append(p)

                    # 基于前景区域计算特征表示
                    if target_percentage == 0:
                        fg_p = torch.zeros(region.size(0), dtype=region.dtype).cuda()
                    else:
                        masked_region = torch.tensor(region_mask[None, :, :]).cuda() * region
                        fg_p = masked_region.sum(axis=(1,2)) / np.sum(region_mask == 1)
                    
                    # 计算背景区域的特征表示
                    if np.sum(region_mask==0) == 0:
                        nfg_p = torch.zeros(region.size(0), dtype=region.dtype).cuda()
                    else:
                        nfg_masked_region = torch.tensor((~region_mask.astype(bool)).astype(int)[None, :, :]).cuda() * region
                        nfg_p = nfg_masked_region.sum(axis=(1,2)) / np.sum(region_mask == 0)

                    patch_locs.append(region_dim_i)

                    fg_pool.append(fg_p.unsqueeze(0))
                    nfg_pool.append(nfg_p.unsqueeze(0))
                    pool_label_list.append(region_label)

        if not pool:
            return [],[],[],[]
        pool=torch.cat(pool,dim=0)
        fg_pool=torch.cat(fg_pool,dim=0)
        nfg_pool=torch.cat(nfg_pool,dim=0)

        return pool, pool_label_list, [fg_score, confidence_score], patch_locs   # 检查patch locs的类型和格式--n*4的python数组


class RoiPoolingRandom(Module):
    def __init__(self, mode='tf', pool_size=(1, 1)):
        """
        tf: (height, width, channels)
        th: (channels, height, width)
        :param mode:
        :param pool_size:
        """
        super(RoiPoolingRandom, self).__init__()

        self.mode = mode
        self.pool_size = pool_size

    # 这里用的是hw的表示，但不影响计算结果，都是全局平均池化
    def pool_region_ori(self, region):

        """
        the pooling of a region
        :param region: the region of interest fetched from feature map
        :return: roipool with size of (1, height, width, channel) if mode is tf otherwise (1, channels, height, width)
        """

        pool_height, pool_width = self.pool_size
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode== 'th':
            region_channels, region_height, region_width = region.shape
            pool = torch.from_numpy(np.zeros((region_channels, pool_height, pool_width)))

        h_step = region_height / pool_height
        w_step = region_width / pool_width
        for i in range(pool_height):
            for j in range(pool_width):

                xmin = j * w_step
                xmax = (j+1) * w_step
                ymin = i * h_step
                ymax = (i+1) * h_step

                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                # if xmin==xmax or ymin==ymax:
                #     continue
                if self.mode=='tf':
                    # pool[i, j, :] = np.max(region[ymin:ymax, xmin:xmax, :], axis=(0,1))
                    pool[i,j,:] = F.adaptive_avg_pool2d(region[ymin:ymax, xmin:xmax, :], (1,1))
                    # pool[i, j, :] = F.adaptive_max_pool2d(region[ymin:ymax, xmin:xmax, :], (1, 1))
                elif self.mode=='th':

                    region_var = region[:, ymin:ymax, xmin:xmax]
                    roi_pooled = F.adaptive_avg_pool2d(region_var, (1, 1))

        return roi_pooled

    def pool_region(self, region):

        """
        the pooling of a region
        :param region: the region of interest fetched from feature map
        :return: roipool with size of (1, height, width, channel) if mode is tf otherwise (1, channels, height, width)
        """

        pool_height, pool_width = self.pool_size
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode== 'th':
            region_channels, region_height, region_width = region.shape
            pool = torch.from_numpy(np.zeros((region_channels, pool_height, pool_width)))

        if self.mode=='th':
            roi_pooled = F.adaptive_avg_pool2d(region, (1 ,1))

        return roi_pooled

    def get_region(self, feature_map, roi_dimensions):
        """
        fetching the roi from feature map by the dimension of the roi
        :param feature_map: the feature map with size of (1, height, width, channels)
        :param roi_dimensions: a region of interest dimensions
        :return:
        """
        xmin, ymin, xmax, ymax = roi_dimensions
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        if self.mode=='tf':
            r = np.squeeze(feature_map)[ymin:ymax, xmin:xmax, :]     # 生成proposal的时候用的是cv2，其xy对应的是横轴和纵轴
        elif self.mode=='th':
            r = feature_map[:, ymin:ymax, xmin:xmax]
        return r
    
    def get_mask_region(self, mask, roi_dimensions):
        xmin, ymin, xmax, ymax = roi_dimensions
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        return mask[ymin:ymax, xmin:xmax]

    def forward(self,feature_map, roi_batch, label_list, patch_nums = [4], norm_cam_bg=[], cls_label=[], area_ratios=[0.2, 0.3, 0.4, 0.5]):
        """
        getting pools from the roi batch
        :param feature_map:
        :param roi_batch: region of interest batch (usually is 256 for faster rcnn)
        :return:
        """
        pool = []
        fg_score = []
        confidence_score = []
        pool_label_list = []
        patch_locs = []
        W, H = feature_map.size(1), feature_map.size(2)

        random_times_base = 10   # 超参
        # iou_threshhold = 0.5  # 超参
        
        cam_predict = np.argmax(norm_cam_bg, 0)

        # 预设面积
        for region_dim, region_label in zip(roi_batch, label_list):
            
            xmin,ymin,xmax,ymax = region_dim
            w, h = xmax-xmin, ymax-ymin
            area = w*h
            
            cur_pic_patches = []
            cur_patches_score = []
            cur_patches_fg_score = []
            cur_patches_confid_score = []
            a_ratio = []

            # # fixed
            # area_ratio = 0.25
            # # mixed
            # area_ratio = random.randint(1,10)/10

            # # 同样筛选掉太小的patch
            # patch_area = area*area_ratio
            # # 原patch面积最小是9*9/4约为20，这里也筛选一下，太小就不要了
            # if patch_area < 10:   # TODO:控制patch最小面积的超参
            #     continue

            # # 根据面积确定w最小比例，同时设置w最大比例
            # range_s, range_l = area_ratio, 1

            random_times = random_times_base
            ppool=[]
            fgpool=[]
            for _ in range(int(random_times)):
                
                # # 随机决定先确定哪一条边
                # ranint = random.randint(0,1)
                # if ranint:
                #     # 在w的设置范围内采样
                #     patch_w = random.randint(round(w*range_s), round(w*range_l))
                #     patch_h = min(round(patch_area/patch_w), h)    # 避免超过proposal大小，用了round可能导致patch_h>h
                # else:
                #     # 在h的设置范围内采样
                #     patch_h = random.randint(round(h*range_s), round(h*range_l))
                #     patch_w = min(round(patch_area/patch_h), w)    # 避免超过proposal大小
                patch_w = int(w/2)
                patch_h = int(h/2)

                patch_iw = random.randint(xmin, int(xmax-patch_w))
                patch_ih = random.randint(ymin, int(ymax-patch_h))


                region_dim_i=[patch_iw, patch_ih, patch_iw+patch_w, patch_ih+patch_h]
                cur_pic_patches.append(region_dim_i)

                # 计算mask
                label_i_mask = np.zeros((W,H))
                label_i_mask[cam_predict == region_label] = 1
                
                # 计算前景占比
                region_mask = self.get_mask_region(label_i_mask, region_dim_i)
                target_percentage = np.sum(region_mask == 1) / np.size(region_mask)
                # # TODO 选fgmid则-0.5后取绝对值再取相反数；选fgfront则直接用target_percentage；选fgback则将target_percentage取相反数
                # cur_patches_score.append(-abs(target_percentage-0.5))
                cur_patches_fg_score.append(target_percentage)
                
                map = feature_map
                region = self.get_region(map, region_dim_i)
                # 基于整个patch计算特征表示
                ppool.append(self.pool_region_ori(region).squeeze().unsqueeze(0))    # [1,channel,1,1]

                # # 基于前景区域计算特征表示
                # if target_percentage == 0:
                #     fgpool.append(torch.zeros(region.size(0), dtype=region.dtype).cuda())
                # else:
                #     masked_region = torch.tensor(region_mask[None, :, :]).cuda() * region
                #     fgpool.append(masked_region.sum(axis=(1,2)) / np.sum(region_mask == 1)) 

                # 计算patch的置信度分数
                c_score = []
                region_cam_predict = self.get_mask_region(cam_predict, region_dim_i)
                label = np.unique(region_cam_predict)  # 0:bg
                for l in label:
                    target_cam_region = self.get_mask_region(norm_cam_bg[l], region_dim_i)
                    c_score.append(target_cam_region[region_cam_predict == l].mean())
                cur_patches_confid_score.append(np.array(c_score).mean())    

            # # NMS
            # b = torch.from_numpy(np.array(cur_pic_patches,np.double)).cuda()
            # s = torch.from_numpy(np.array(cur_patches_score,np.double)).cuda()
            # bounding_box_index = torchvision.ops.nms(b,s,iou_threshhold).cpu()
            # 检查一下NMS后的patches的面积占比（会不会小面积的都被大面积的抑制掉了）--不会，反而小尺寸的多 eg: 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3
            # print(np.array(a_ratio)[bounding_box_index])

            # # === 统计随机生成的10个patch的分布情况 =====
            # patches_start = np.array([[(temp[0]-xmin)/w,(temp[1]-ymin)/h] for temp in cur_pic_patches])
            # patches_center = np.mean(patches_start, axis=0)
            # # patches_center = np.tile(patches_center, (patches_start.shape[0], 1))
            # distances = np.sqrt(np.sum((patches_start-patches_center)**2, axis=1))
            # norm_mean_dist = np.mean(distances)
            # device = torch.cuda.current_device()
            # with open(f"/usr/volume/WSSS/wsss_pml/distances_{device}.txt", "a") as f:
            #     f.write(f"{norm_mean_dist:.4f}\n") 
            # # === 统计随机生成的10个patch的分布情况 =====

            # noNMS
            bounding_box_index = range(len(cur_pic_patches))

            for i in bounding_box_index:

                region_dim_i = cur_pic_patches[i]

                pool.append(ppool[i])
                # fg_pool.append(fgpool[i].unsqueeze(0))
                patch_locs.append(region_dim_i)
                pool_label_list.append(region_label)
                fg_score.append(cur_patches_fg_score[i])
                confidence_score.append(cur_patches_confid_score[i])

        # if not fg_pool:
        #     return [],[],[],[],[],[]
        if not pool:
            return [],[],[],[],[],[]

        pool=torch.cat(pool,dim=0)
        # fg_pool=torch.cat(fg_pool,dim=0)

        return pool, pool_label_list, [fg_score, confidence_score], patch_locs   # 检查patch locs的类型和格式--n*4的python数组

class RoiPoolingContrastive(Module):
    def __init__(self, mode='tf', pool_size=(1, 1), args=""):
        """
        tf: (height, width, channels)
        th: (channels, height, width)
        :param mode:
        :param pool_size:
        """
        super(RoiPoolingContrastive, self).__init__()

        self.mode = mode
        self.pool_size = pool_size
        self.crop = ContrastiveCrop(alpha=args.ccrop_alpha)

    # 这里用的是hw的表示，但不影响计算结果，都是全局平均池化
    def pool_region_ori(self, region):

        """
        the pooling of a region
        :param region: the region of interest fetched from feature map
        :return: roipool with size of (1, height, width, channel) if mode is tf otherwise (1, channels, height, width)
        """

        pool_height, pool_width = self.pool_size
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode== 'th':
            region_channels, region_height, region_width = region.shape
            pool = torch.from_numpy(np.zeros((region_channels, pool_height, pool_width)))

        h_step = region_height / pool_height
        w_step = region_width / pool_width
        for i in range(pool_height):
            for j in range(pool_width):

                xmin = j * w_step
                xmax = (j+1) * w_step
                ymin = i * h_step
                ymax = (i+1) * h_step

                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                # if xmin==xmax or ymin==ymax:
                #     continue
                if self.mode=='tf':
                    # pool[i, j, :] = np.max(region[ymin:ymax, xmin:xmax, :], axis=(0,1))
                    pool[i,j,:] = F.adaptive_avg_pool2d(region[ymin:ymax, xmin:xmax, :], (1,1))
                    # pool[i, j, :] = F.adaptive_max_pool2d(region[ymin:ymax, xmin:xmax, :], (1, 1))
                elif self.mode=='th':

                    region_var = region[:, ymin:ymax, xmin:xmax]
                    roi_pooled = F.adaptive_avg_pool2d(region_var, (1, 1))

        return roi_pooled

    def pool_region(self, region):

        """
        the pooling of a region
        :param region: the region of interest fetched from feature map
        :return: roipool with size of (1, height, width, channel) if mode is tf otherwise (1, channels, height, width)
        """

        pool_height, pool_width = self.pool_size
        if self.mode == 'tf':
            region_height, region_width, region_channels = region.shape
            pool = np.zeros((pool_height, pool_width, region_channels))
        elif self.mode== 'th':
            region_channels, region_height, region_width = region.shape
            pool = torch.from_numpy(np.zeros((region_channels, pool_height, pool_width)))

        if self.mode=='th':
            roi_pooled = F.adaptive_avg_pool2d(region, (1 ,1))

        return roi_pooled

    def get_region(self, feature_map, roi_dimensions):
        """
        fetching the roi from feature map by the dimension of the roi
        :param feature_map: the feature map with size of (1, height, width, channels)
        :param roi_dimensions: a region of interest dimensions
        :return:
        """
        xmin, ymin, xmax, ymax = roi_dimensions
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        if self.mode=='tf':
            r = np.squeeze(feature_map)[ymin:ymax, xmin:xmax, :]     # 生成proposal的时候用的是cv2，其xy对应的是横轴和纵轴
        elif self.mode=='th':
            r = feature_map[:, ymin:ymax, xmin:xmax]
        return r
    
    def get_mask_region(self, mask, roi_dimensions):
        xmin, ymin, xmax, ymax = roi_dimensions
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        return mask[ymin:ymax, xmin:xmax]

    def forward(self,feature_map, roi_batch, label_list, patch_nums = [4], norm_cam_bg=[], cls_label=[], area_ratios=[0.2, 0.3, 0.4, 0.5]):
        """
        getting pools from the roi batch
        :param feature_map:
        :param roi_batch: region of interest batch (usually is 256 for faster rcnn)
        :return:
        """
        pool = []
        fg_pool = []
        fg_score = []
        pool_label_list = []
        patch_locs = []
        W, H = feature_map.size(1), feature_map.size(2)
        confidence_score = []

        random_times_base = 10   # 超参
        # iou_threshhold = 0.5  # 超参
        
        cam_predict = np.argmax(norm_cam_bg, 0)

        # 预设面积
        for region_dim, region_label in zip(roi_batch, label_list):
            
            xmin,ymin,xmax,ymax = region_dim
            w, h = xmax-xmin, ymax-ymin
            area = w*h
            
            cur_pic_patches = []
            cur_patches_score = []
            cur_patches_fg_score = []
            a_ratio = []

            # fixed
            area_ratio = 0.25
            # mixed
            # area_ratio = random.randint(1,10)/10

            random_times = random_times_base
            ppool=[]
            fgpool=[]
            for _ in range(int(random_times)):
                
                patch_iw, patch_ih, patch_w, patch_h = self.crop(feature_map[0], region_dim)

                region_dim_i=[patch_iw, patch_ih, patch_iw+patch_w, patch_ih+patch_h]
                cur_pic_patches.append(region_dim_i)

                # 计算mask
                label_i_mask = np.zeros((W,H))
                label_i_mask[cam_predict == region_label] = 1
                # 计算前景占比
                region_mask = self.get_mask_region(label_i_mask, region_dim_i)
                target_percentage = np.sum(region_mask == 1) / np.size(region_mask)

                # # 选fgmid则-0.5后取绝对值再取相反数；选fgfront则直接用target_percentage；选fgback则将target_percentage取相反数
                # cur_patches_score.append(-abs(target_percentage-0.5))
                cur_patches_fg_score.append(target_percentage)

                map = feature_map
                region = self.get_region(map, region_dim_i)
                # 基于整个patch计算特征表示
                ppool.append(self.pool_region_ori(region).squeeze().unsqueeze(0))    # [1,channel,1,1]

                # # 基于前景区域计算特征表示
                # if target_percentage == 0:
                #     fgpool.append(torch.zeros(region.size(0), dtype=region.dtype).cuda())
                # else:
                #     masked_region = torch.tensor(region_mask[None, :, :]).cuda() * region
                #     fgpool.append(masked_region.sum(axis=(1,2)) / np.sum(region_mask == 1)) 

                # 计算patch的置信度分数
                # patch_cam = self.get_mask_region(norm_cam[region_label-1], region_dim_i)
                # confidence_score.append(patch_cam.mean())    

            # # NMS
            # b = torch.from_numpy(np.array(cur_pic_patches,np.double)).cuda()
            # s = torch.from_numpy(np.array(cur_patches_score,np.double)).cuda()
            # bounding_box_index = torchvision.ops.nms(b,s,iou_threshhold).cpu()
            # 检查一下NMS后的patches的面积占比（会不会小面积的都被大面积的抑制掉了）--不会，反而小尺寸的多 eg: 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3
            # print(np.array(a_ratio)[bounding_box_index])
            # noNMS
            bounding_box_index = range(len(cur_pic_patches))

            for i in bounding_box_index:

                region_dim_i = cur_pic_patches[i]
                pool.append(ppool[i])
                # fg_pool.append(fgpool[i].unsqueeze(0))
                patch_locs.append(region_dim_i)
                pool_label_list.append(region_label)
                fg_score.append(cur_patches_fg_score[i])

        if not pool:
            return [],[],[],[],[],[]
        # if not fg_pool:
        #     return [],[],[],[],[],[]

        pool=torch.cat(pool,dim=0)
        # fg_pool=torch.cat(fg_pool,dim=0)

        return pool, pool_label_list, fg_score, patch_locs   # 检查patch locs的类型和格式--n*4的python数组
