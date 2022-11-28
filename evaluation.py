import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
from voc12.data import get_img_path
from tool.visualization import color_pro

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
# categories = ['0-background','1aeroplane','2bicycle','3bird','4boat','5bottle','6bus','7car','8cat','9chair','10cow',
#                 '11diningtable','12dog','13horse','14motorbike','15person','16pottedplant','17sheep','18sofa','19train','20tvmonitor']
def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, bg_thres=0.15):
    TP = []
    P = []
    T = []
    ALL = []
    # IOU = [] # 用IoU作为指标选择预测结果进行可视化，需要记录所有图片的IoU
    # 上锁
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
        ALL.append(multiprocessing.Value('i', 0, lock=True))
        # IOU.append(multiprocessing.Array('i', [], lock=True))
    
    def compare(start,step,TP,P,T,ALL):
        for idx in range(start,len(name_list),step):
            #print('%d/%d'%(idx,len(name_list)))
            name = name_list[idx]
            predict_file = os.path.join(predict_folder,'%s.png'%name)
            predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            cal = gt<255
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)   # 预测成该类别的区域大小，不算白边
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)    # 真实是该类别的区域大小，这里*cal不算白边，但本来也不会包括白边的吧
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)   # 真实是该类别的区域中，预测也是该类别的区域大小，这里mask也有*cal，但感觉可以不用
                TP[i].release()
                ALL[i].acquire()
                ALL[i].value += predict.size
                ALL[i].release()
            
            # gt_classes = np.unique(gt)[1:-1]  # 去掉每张图片中的白边（255）和黑色背景（0）
            # for i in gt_classes:
            #     inter = np.sum((gt==i)*mask)
            #     iou = inter / (np.sum((predict==i)*cal) + np.sum((gt==i)*cal) - inter + 1e-10)
            #     IOU[i-1].acquire()
            #     IOU[i-1].value.append((iou, name))
            #     IOU[i-1].release()


    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,ALL))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    # ACC = []
    RECALL = []
    PRECISION = []
    # FP = []
    # FN = []
    # TN = []
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        RECALL.append(TP[i].value/(T[i].value+1e-10))   # 即TPR
        PRECISION.append(TP[i].value/(P[i].value+1e-10))
        
        # # 没有意义 TN比重太大，同理FPR也没意义，即涉及到TN的评估标准都没意义
        # ACC.append((ALL[i].value-T[i].value-P[i].value+2*TP[i].value)/(ALL[i].value+1e-10))
        # TN.append((ALL[i].value-T[i].value-P[i].value+TP[i].value)/(ALL[i].value+1e-10))    
        # FP.append((P[i].value-TP[i].value)/(ALL[i].value+1e-10))
        # FN.append((T[i].value-TP[i].value)/(ALL[i].value+1e-10))

        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
               
    miou = np.mean(np.array(IoU))
    miou_foreground = np.mean(np.array(IoU)[1:])
    # macc = np.mean(np.array(ACC))
    # macc_foreground = np.mean(np.array(ACC)[1:])
    mrecall = np.mean(np.array(RECALL))
    mrecall_foreground = np.mean(np.array(RECALL)[1:])
    mprecision = np.mean(np.array(PRECISION))
    mprecision_foreground = np.mean(np.array(PRECISION)[1:])
    # mtp = np.mean(np.array([TP[i].value/(ALL[i].value+1e-10) for i in range(num_cls)]))
    # mfp = np.mean(np.array(FP))
    # mfn = np.mean(np.array(FN))
    # mtn = np.mean(np.array(TN))
    t_tp = np.mean(np.array(T_TP)[1:])
    p_tp = np.mean(np.array(P_TP)[1:])
    fp_all = np.mean(np.array(FP_ALL)[1:])
    fn_all = np.mean(np.array(FN_ALL)[1:])
    print('\n======================================================')
    print('%11s:%7.3f%%'%('mIoU',miou*100))
    # print('%11s:%7.3f'%('T/TP',t_tp))
    # print('%11s:%7.3f'%('P/TP',p_tp))
    # print('%11s:%7.3f'%('FP/ALL',fp_all))
    # print('%11s:%7.3f'%('FN/ALL',fn_all))
    # print('%11s:%7.3f'%('miou_foreground',miou_foreground))
    loglist['mIoU'] = '%.4f%%'%(miou*100)
    loglist['miou_foreground'] = "{:.4f}".format(miou_foreground) 
    # loglist['macc'] = "{:.2f}".format(macc) 
    # loglist['macc_foreground'] = "{:.2f}".format(macc_foreground) 
    loglist['mrecall'] = "{:.4f}".format(mrecall) 
    loglist['mrecall_foreground'] = "{:.4f}".format(mrecall_foreground) 
    loglist['mprecision'] = "{:.4f}".format(mprecision) 
    loglist['mprecision_foreground'] = "{:.4f}".format(mprecision_foreground) 
    # loglist['TP'] = "{:.2f}".format(mtp) 
    # loglist['FP'] = "{:.2f}".format(mfp) 
    # loglist['TN'] = "{:.2f}".format(mtn) 
    # loglist['FN'] = "{:.2f}".format(mfn) 

    loglist['t_tp'] = t_tp
    loglist['p_tp'] = p_tp
    loglist['fp_all'] = "{:.2f}".format(fp_all)
    loglist['fn_all'] = "{:.2f}".format(fn_all)

    detaillist = {"aThreshold": bg_thres, 
        "bMean IoU": "{:.4f}".format(miou),
        "cMean IoU FG": "{:.4f}".format(miou_foreground),
        # "dMean Acc": "{:.2f}".format(macc),
        # "eMean Acc FG": "{:.2f}".format(macc_foreground),
        "fMean_Recall": "{:.4f}".format(mrecall),
        "gMean_Recall FG": "{:.4f}".format(mrecall_foreground),
        "hMean_Precision": "{:.4f}".format(mprecision),
        "iMean_Precision FG": "{:.4f}".format(mprecision_foreground),
        "jIoU": ["{:.2f}".format(i) for i in IoU],
        # "kAcc": ["{:.2f}".format(i) for i in ACC],
        "lRecall": ["{:.2f}".format(i) for i in RECALL],
        "mPrecision": ["{:.2f}".format(i) for i in PRECISION],
        # "nTP": ["{:.2f}".format(i) for i in [(TP[i].value)/(ALL[i].value+1e-10) for i in range(num_cls)]],
        # "oFP": ["{:.2f}".format(i) for i in FP],
        # "pTN": ["{:.2f}".format(i) for i in TN],
        # "qFN": ["{:.2f}".format(i) for i in FN],
        "nFP_ALL": ["{:.2f}".format(i) for i in FP_ALL],
        "oFN_ALL": ["{:.2f}".format(i) for i in FN_ALL],
        }
    
    return loglist, detaillist

def do_visualization(predict_folder, gt_folder, name_list, num_cls=20, visualize_dir="", cams_dir=""):

    if not os.path.exists(visualize_dir):
        os.mkdir(visualize_dir)

    result = []
    for i in range(num_cls):
        result.append([])

    for idx in range(len(name_list)):
        name = name_list[idx]
        predict_file = os.path.join(predict_folder,'%s.png'%name)
        predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)

        gt_file = os.path.join(gt_folder,'%s.png'%name)
        gt = np.array(Image.open(gt_file))
        cal = gt<255
        mask = (predict==gt) * cal
    
        gt_classes = np.unique(gt)[1:-1]  # 去掉每张图片中的白边（255）和黑色背景（0）
        for i in gt_classes:
            inter = np.sum((gt==i)*mask)
            iou = inter / (np.sum((predict==i)*cal) + np.sum((gt==i)*cal) - inter + 1e-10)
            result[i-1].append((iou, name))

    # visualize
    visualize_num = 5
    for i in range(num_cls):
        if len(result[i])<2*visualize_num:
            continue
        iou_list = [temp[0] for temp in result[i]]
        name_list = [temp[1] for temp in result[i]]
        sorted_idxs = np.argsort(np.array(iou_list))   # 升序排序
        fronts = sorted_idxs[-visualize_num:]
        backs = sorted_idxs[:visualize_num]
        for j_idx, j in enumerate(fronts):
            save_dir = visualize_dir + "/{}_rank{}_{}_score{:.2f}.png".format(categories[i+1], visualize_num-j_idx, name_list[j], iou_list[j])
            img = np.array(Image.open(get_img_path(name_list[j], "/usr/volume/WSSS/VOCdevkit/VOC2012")))  # 检查是否为hwc--yes
            cams = np.load(os.path.join(cams_dir, name_list[j] + '.npy'), allow_pickle=True).item()  # 保存前已被resize到跟img一样大，同时砍去负值且进行了规范化
            if i in cams:
                cam = cams[i]
            else:
                continue
            cam_img = color_pro(cam, img)
            # 检查数值的大小--0-255  chw或者hwc--hwc
            Image.fromarray(cam_img).save(save_dir)

        for j_idx, j in enumerate(backs):
            save_dir = visualize_dir + "/{}_rank-{}_{}_score{:.2f}.png".format(categories[i+1], j_idx+1, name_list[j], iou_list[j])
            img = np.array(Image.open(get_img_path(name_list[j], "/usr/volume/WSSS/VOCdevkit/VOC2012")))  # 检查是否为hwc
            cams = np.load(os.path.join(cams_dir, name_list[j] + '.npy'), allow_pickle=True).item()  # 保存前已被resize到跟img一样大，同时砍去负值且进行了规范化
            if i in cams:
                cam = cams[i]
            else:
                continue
            cam_img = color_pro(cam, img)
            # 检查数值的大小  hwc
            Image.fromarray(cam_img).save(save_dir)

def writedictJson(path, dictionary):
    with open(path, "a") as f:
        json.dump(dictionary, f, indent=4, sort_keys=True)

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        if key == 'TP':
            s += '\n'
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment, bg_thres):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    logfile.write('background_threshold:%s\n'%bg_thres)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()

def eval(voc_list_txt, pred_dir, gt_dir='/usr/volume/WSSS/VOCdevkit/VOC2012/SegmentationClass',
         saved_txt="",
         detail_txt="",
         visualize_dir="",
         cams_dir="",
         model_name="",
         num_class=21,
         bg_threshold=0.15):
    df = pd.read_csv(voc_list_txt, names=['filename'])
    name_list = df['filename'].values
    # do_visualization(pred_dir, gt_dir, name_list, num_cls=20, visualize_dir=visualize_dir, cams_dir=cams_dir)
    loglist, detaillist = do_python_eval(pred_dir, gt_dir, name_list, num_class, bg_threshold)  # pred_dir预测的CAMs，gt_dir所有图片的真实标签的位置，name_list评估所用的图片的名字清单
    writelog(saved_txt, loglist, model_name, bg_threshold)
    writedictJson(detail_txt, detaillist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_dir", default='/usr/volume/WSSS/WSSS_PML/result/e5-patch_weight0.05/out_cam/0.23', type=str)

    parser.add_argument("--list", default='/usr/volume/WSSS/WSSS_PML/voc12/val.txt', type=str)
    parser.add_argument("--gt_dir", default='/usr/volume/WSSS/VOCdevkit/VOC2012/SegmentationClass', type=str)
    parser.add_argument('--logfile', default='/usr/volume/WSSS/WSSS_PML/result/e5-patch_weight0.05/saved_checkpoints/log_txt/evallog.txt',type=str)
    parser.add_argument('--logfile_detail', default='/usr/volume/WSSS/WSSS_PML/result/e5-patch_weight0.05/saved_checkpoints/log_txt/evallog1.txt',type=str)
    parser.add_argument('--comment', default='', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    # name_list = name_list[np.random.choice(range(len(name_list)), 100)]  # 只取一部分，加快速度
    # bg_thres = 0.15
    bg_thresh=[0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30]
    visualize_dir = "/usr/volume/WSSS/WSSS_PML/result/e5-patch_weight0.05-all-patchscale4-confidback0.5-nohards1/visualization/epoch0"
    cams_dir="/usr/volume/WSSS/WSSS_PML/result/e5-patch_weight0.05-all-patchscale4-confidback0.5-nohards1/out_cams"
    pre_dir_root = "/usr/volume/WSSS/WSSS_PML/result/e5-patch_weight0.05-all-patchscale4-confidback0.5-nohards1/out_cam_pre/epoch0"
    # loglist, detaillist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21, bg_thres)
    # writelog(args.logfile, loglist, args.comment, bg_thres)
    # writedictJson(args.logfile_detail, detaillist)

    for bgt in bg_thresh:

        cur_pre_dir = pre_dir_root + f"/{bgt}"
        cur_vis_dir = visualize_dir + f"/{bgt}"
        if not os.path.exists(cur_vis_dir):
            os.mkdir(cur_vis_dir)
        
        do_visualization(cur_pre_dir, args.gt_dir, name_list, 20, cur_vis_dir, cams_dir)
