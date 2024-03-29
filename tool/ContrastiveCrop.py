from torchvision.transforms import RandomResizedCrop
import torch
import random
import numpy as np
import math
from torch.distributions.beta import Beta
import torchvision.transforms.functional as F
from torch.nn.modules.module import Module

class ContrastiveCrop(Module):  # adaptive beta
    def __init__(self, alpha=1.0, **kwargs):
        super(ContrastiveCrop, self).__init__()
        # a == b == 1.0 is uniform distribution
        self.beta = Beta(alpha, alpha)
        self.scale=(0.25, 0.25)
        self.ratio=(1. / 1., 1. / 1.)
        # self.scale=(0.2, 0.8)
        # self.ratio=(3. / 4., 4. / 3.)

    def get_params(self, img, box, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        # width, height = F._get_image_size(img)
        width, height = img.size()  # width，height分别对应纵轴，横轴
        area = height * width
        x0, y0, x1, y1 = box   # xy分别对应横轴纵轴
        box_area = (x1-x0)*(y1-y0)

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = box_area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                
                ch0 = min(max(x0 - h//2, 0), height - h)
                ch1 = min(max(x1 - h//2, 0), height - h)
                cw0 = min(max(y0 - w//2, 0), width - w)
                cw1 = min(max(y1 - w//2, 0), width - w)

                i = ch0 + int((ch1 - ch0) * self.beta.sample())
                j = cw0 + int((cw1 - cw0) * self.beta.sample())
                return i, j, h, w    #起点，宽高

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img, box):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, box, self.scale, self.ratio)
        return i, j, h, w

