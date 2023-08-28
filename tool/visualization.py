import os

import numpy as np
import torch
import torch.nn.functional as F
import cv2
# from cv2.ximgproc import l0Smooth
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import numpy as np
from sklearn.manifold import TSNE
import PIL.Image
import matplotlib.pyplot as plt
from voc12.data import get_img_path
from tool.imutils import reNormalize


# from tool import pyutils
# seed = pyutils.seed_everything()



def color_pro(pro, img=None, rate=0.5, mode='hwc'):
	H, W = pro.shape
	pro_255 = (pro*255).astype(np.uint8)
	pro_255 = np.expand_dims(pro_255,axis=2)
	color = cv2.applyColorMap(pro_255,cv2.COLORMAP_JET)
	color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
	if img is not None:
		if mode == 'hwc':
			assert img.shape[0] == H and img.shape[1] == W
			color = cv2.addWeighted(img,rate,color,1-rate,0)
		elif mode == 'chw':
			assert img.shape[1] == H and img.shape[2] == W
			img = np.transpose(img,(1,2,0))
			color = cv2.addWeighted(img,rate,color,1-rate,0)
			color = np.transpose(color,(2,0,1))
	else:
		if mode == 'chw':
			color = np.transpose(color,(2,0,1))	
	return color
		
def generate_vis(p, gt, img, func_label2color, threshold=0.1, norm=True):
	# All the input should be numpy.array 
	# img should be 0-255 uint8
	C, H, W = p.shape            # DONOTUNDERSTAND:gt[0]=gt[1]=1其他都是0，但是p[1]全为0（p[0]是因为初始化为0），其他的有0有非0

	if norm:
		prob = max_norm(p, 'numpy')   # 规范化
	else:
		prob = p
	prob = prob * gt     # 将图片中没有的类的cam中的激活值全赋为0
	prob[prob<=0] = 1e-7
	if threshold is not None:
		prob[0,:,:] = np.power(1-np.max(prob[1:,:,:],axis=0,keepdims=True), 4)      # 计算背景的激活值

	CLS = ColorCLS(prob, func_label2color)	
	CAM = ColorCAM(prob, img)

	prob_crf = dense_crf(prob, img, n_classes=C, n_iters=1)
	
	CLS_crf = ColorCLS(prob_crf, func_label2color)
	CAM_crf = ColorCAM(prob_crf, img)
	
	return CLS, CAM, CLS_crf, CAM_crf


def generate_vis_for_seed_areas(p, img, func_label2color, weight=0.8, threshold=0.1, norm=False):
	# All the input should be numpy.array
	# img should be 0-255 uint8
	C, H, W = p.shape

	if norm:
		prob = max_norm(p, 'numpy')
	else:
		prob = p

	prob[prob <= 0] = 1e-7
	prob=np.max(prob, axis=0, keepdims=True)

	CAM=color_pro(prob[0, :, :], img=img, rate=weight, mode='chw')

	return CAM

def generate_vis_for_aff(p, img, func_label2color, threshold=0.1, norm=False):
	# All the input should be numpy.array
	# img should be 0-255 uint8
	C, H, W = p.shape

	if norm:
		prob = max_norm(p, 'numpy')
	else:
		prob = p

	prob[prob <= 0] = 1e-7
	# if threshold is not None:
	# 	prob[0, :, :] = np.power(1 - np.max(prob[1:, :, :], axis=0, keepdims=True), 4)

	CLS = ColorCLS(prob, func_label2color)
	CAM = ColorCAM(prob, img)

	prob_crf = dense_crf(prob, img, n_classes=C, n_iters=1)

	CLS_crf = ColorCLS(prob_crf, func_label2color)
	CAM_crf = ColorCAM(prob_crf, img)

	return CLS, CAM, CLS_crf, CAM_crf

def max_norm(p, version='torch', e=1e-5):
	if version is 'torch':
		if p.dim() == 3:
			C, H, W = p.size()
			max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)+e
			p = F.relu(p-e, inplace=True)
			p = p/max_v
		elif p.dim() == 4:
			N, C, H, W = p.size()
			max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)+e
			p = F.relu(p-e, inplace=True)
			p = p/max_v
	elif version is 'numpy' or version is 'np':
		if p.ndim == 3:
			C, H, W = p.shape
			max_v = np.max(p,(1,2),keepdims=True)+e
			p -= e
			p[p<0] = 0
			p = p/max_v
		elif p.ndim == 4:
			N, C, H, W = p.shape
			max_v = np.max(p,(2,3),keepdims=True)+e
			p -= e
			p[p<0] = 0
			p = p/max_v
	return p

def ColorCAM(prob, img):
	assert prob.ndim == 3
	C, H, W = prob.shape
	colorlist = []
	for i in range(C):
		colorlist.append(color_pro(prob[i,:,:], img=img, mode='chw'))
	CAM = np.array(colorlist)/255.0
	return CAM
	
def ColorCLS(prob, func_label2color):
	assert prob.ndim == 3
	prob_idx = np.argmax(prob, axis=0)
	CLS = func_label2color(prob_idx).transpose((2,0,1))
	return CLS
	
# 将单通道的标签图变成三通道RGB标签图
def VOClabel2colormap(label):
	m = label.astype(np.uint8)
	r,c = m.shape
	cmap = np.zeros((r,c,3), dtype=np.uint8)
	cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
	cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
	cmap[:,:,2] = (m&4)<<5
	cmap[m==255] = [255,255,255]
	return cmap

def dense_crf(probs, img=None, n_classes=21, n_iters=1, scale_factor=1):
	c,h,w = probs.shape

	if img is not None:
		assert(img.shape[1:3] == (h, w))
		img = np.transpose(img,(1,2,0)).copy(order='C')

	d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
	
	unary = unary_from_softmax(probs)
	unary = np.ascontiguousarray(unary)
	d.setUnaryEnergy(unary)
	d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
	d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
	Q = d.inference(n_iters)

	preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w))
	return preds

def gen_cam_vis():
	cam_root = "/usr/volume/WSSS/wsss_pml/out_cam_train"
	cam_saved_root = "/usr/volume/WSSS/wsss_pml/visualization_results_w6/cam_pml/"
	os.makedirs(cam_saved_root, exist_ok=True)
	voc_root = "/usr/volume/WSSS/VOCdevkit/VOC2012"
	with open("/usr/volume/WSSS/wsss_pml/voc12/train_voc12.txt") as fp:
		lines = fp.readlines()
	for idx, line in enumerate(lines):
		# line="/JPEGImages/2007_000392.jpg /SegmentationClass/2007_000392.png"
		line = line.strip().split(" ")
		img_id = line[1][-15:-4]
		img_path = voc_root + line[0]
		gt_path = voc_root + line[1]
		cam_path = os.path.join(cam_root, f"{img_id}.npy")

		cam = np.asarray(list(np.load(cam_path, allow_pickle=True).item().values()))  # .transpose(1, 2, 0)
		# img_8=img = np.asarray(PIL.Image.open(img_path).convert("RGB")).transpose((2, 0, 1))
		img_8 = img = np.asarray(PIL.Image.open(img_path).convert("RGB")).transpose((2, 0, 1))

		CAM = generate_vis_for_seed_areas(cam, img_8, func_label2color=VOClabel2colormap, weight=0.6).transpose(1, 2, 0)
		CAM = cv2.cvtColor(CAM, cv2.COLOR_BGR2RGB)
		cv2.imwrite(f"{cam_saved_root}/{img_id}.png", CAM)

		print(idx)
		temp = 0

def gen_pseudo_mask_vis():
	pseudo_mask_saved_root = "/home/chenkeke/project/WSSS/psa/visualization_clm/"
	os.makedirs(pseudo_mask_saved_root, exist_ok=True)
	pseudo_mask_root = "/home/chenkeke/project/WSSS/psa/out_cam_pred/"

	with open("/usr/volume/WSSS/wsss_pml/voc12/train_voc12.txt") as fp:
		lines = fp.readlines()
	for idx, line in enumerate(lines):

		line = line.strip().split(" ")
		img_id = line[1][-15:-4]
		img_path = pseudo_mask_root + img_id + ".png"

		pseudo_mask = np.asarray(PIL.Image.open(img_path))

		CLS = VOClabel2colormap(pseudo_mask)

		CLS = cv2.cvtColor(CLS, cv2.COLOR_BGR2RGB)
		cv2.imwrite(f"{pseudo_mask_saved_root}/{img_id}.png", CLS)

		print(idx)

def gen_object_proposals():
	pass

def visualize_patch(patches, patch_labels, patch_mask, save_dir, epoch):
	# tSNE降维以及可视化
	tsne = TSNE(n_components=2, learning_rate='auto').fit_transform(patches)
	plt.figure(figsize=(40,40), dpi=120)
	# 被选择的patches
	if len(patch_labels[patch_mask==1]) > 0:
		plt.scatter(tsne[patch_mask==1,0], tsne[patch_mask==1,1], c=patch_labels[patch_mask==1], marker='+', cmap=plt.cm.Spectral, edgecolors='k', alpha=1, s=100)
	# 没有被选择的patches
	if len(patch_labels[patch_mask==0]) > 0:
		scatter = plt.scatter(tsne[patch_mask==0,0], tsne[patch_mask==0,1], c=patch_labels[patch_mask==0], marker='o', cmap=plt.cm.Spectral, alpha=0.4)
		plt.legend(handles=scatter.legend_elements(num=None)[0], labels=[f'{i}' for i in range(1,21)], loc='best')
	plt.savefig(save_dir + f'epoch{epoch}_visualize_patches-s.jpg')

	plt.figure(figsize=(20,20), dpi=80)
	if len(patch_labels[patch_mask==1]) > 0:
		plt.scatter(tsne[patch_mask==1,0], tsne[patch_mask==1,1], c=patch_labels[patch_mask==1], marker='+', cmap=plt.cm.Spectral, edgecolors='k', alpha=1, s=100)
	if len(patch_labels[patch_mask==0]) > 0:
		scatter = plt.scatter(tsne[patch_mask==0,0], tsne[patch_mask==0,1], c=patch_labels[patch_mask==0], marker='o', cmap=plt.cm.Spectral, alpha=0.4)
		plt.legend(handles=scatter.legend_elements(num=None)[0], labels=[f'{i}' for i in range(1,21)], loc='best')
	plt.savefig(save_dir + f'epoch{epoch}_visualize_patches-b.jpg')

'''
categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
def visualize_batch_patches(imgs, img_names, patch_locs, patch_img_labels, patch_labels, patch_mask, patch_scores, triples, args, epoch_iter):

	device = torch.cuda.current_device()
	if device==1:
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
	
	for i in range(len(imgs)):
		cur_img = PIL.Image.open(get_img_path(img_names[i], args.voc12_root)).convert("RGB")
		cur_img.save(unselected_dir+'/img{}.png'.format(img_names[i]))

	unselected_idxs = np.array(range(len(patch_mask)))[patch_mask == 0]
	for i in range(len(unselected_idxs)):
		origin_idx = unselected_idxs[i]
		cur_img = img_list[patch_img_labels[origin_idx]]
		patch = cur_img.crop(patch_locs[origin_idx])
		size = patch.size
		patch = patch.resize((300, int(300/size[0]*size[1])))
		patch.save(unselected_dir+'/img{}_{}_score{:.3f}.png'.format(img_names[patch_img_labels[origin_idx]], categories[patch_labels[origin_idx]-1], patch_scores[origin_idx]))

	selected_idxs = np.array(range(len(patch_mask)))[patch_mask == 1]
	for i in range(len(selected_idxs)):
		origin_idx = selected_idxs[i]
		cur_img = img_list[patch_img_labels[origin_idx]]
		patch = cur_img.crop(patch_locs[origin_idx])
		size = patch.size
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
'''	

if __name__ == "__main__":
	# cam_root="/usr/volume/WSSS/wsss_pml/out_cam_train"
	pseudo_mask_saved_root="/home/chenkeke/project/WSSS/psa/visualization_proposals/"
	os.makedirs(pseudo_mask_saved_root, exist_ok=True)
	pseudo_mask_root="/home/chenkeke/project/WSSS/psa/out_cam_pred/"
	voc_root = "/usr/volume/WSSS/VOCdevkit/VOC2012"

	with open("/usr/volume/WSSS/wsss_pml/voc12/train_voc12.txt") as fp:
		lines=fp.readlines()
	for idx, line in enumerate(lines):
		# line="/JPEGImages/2007_000392.jpg /SegmentationClass/2007_000392.png"
		line=line.strip().split(" ")
		img_id = line[1][-15:-4]
		img_path=pseudo_mask_root+img_id+".png"

		pseudo_mask = np.asarray(PIL.Image.open(img_path))

		img_8 = img = np.asarray(PIL.Image.open(voc_root + line[0]).convert("RGB"))#.transpose((2, 0, 1))
		img_8=cv2.cvtColor(img_8, cv2.COLOR_BGR2RGB)

		CLS = VOClabel2colormap(pseudo_mask)  # .transpose((2,0,1))
		CLS = cv2.cvtColor(CLS, cv2.COLOR_BGR2RGB)


		# contours, hier = cv2.findContours(pseudo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# for c in contours:
		# 	x, y, w, h = cv2.boundingRect(c)
		# 	if w < 20 or h < 20 or (h / w) > 4 or (w / h) > 4:  # filter too small bounding box
		# 		continue
		# 	img_8=cv2.rectangle(img_8, (x, y), (x+w, y+h), (0, 0, 255), 2)
		# 	CLS=cv2.rectangle(CLS, (x, y), (x+w, y+h), (0, 0, 255), 2)


		# CLS = ColorCLS(pseudo_mask, func_label2color=VOClabel2colormap)

		# cv2.imwrite(f"{pseudo_mask_saved_root}/{img_id}_img.png", img_8)
		cv2.imwrite(f"{pseudo_mask_saved_root}/{img_id}_clm.png", CLS)

		print(idx)
		temp=0




