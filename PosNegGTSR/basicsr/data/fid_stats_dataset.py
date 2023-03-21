import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from torch.utils import data as data
import json
from collections import Counter
from torchvision.transforms.functional import normalize

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, imfromfile
from basicsr.utils.registry import DATASET_REGISTRY

def check_list(*l):
    count = Counter(l)
    p_count = count['Positive']
    s_count = count['Similar']
    n_count = count['Negative']
    return p_count, s_count, n_count

def find_max_repetition(*l):
    count = Counter(l)
    most_counterNum = count.most_common(1)
    most_element = most_counterNum[0][0]
    return most_element

@DATASET_REGISTRY.register()
class Fid_StatsDataset(data.Dataset):
    def __init__(self, opt):
        super(Fid_StatsDataset, self).__init__()
        self.opt = opt
        self.all_gt_folder = opt['dataroot_all_gt']  # 0 - 20000 sets of images folder. The folder has five sub folders, i.e. original, 01, 02, 03, 04, 05, each sub folder contains different GTs.
        self.pos_img_list_each = opt['pos_img_list_each']  # 0 - 20000 labels, A, B, C sub folders, represents one set of images has three different labels from three different people.
        self.enhanced_num = opt['enhanced_num']
        self.mean = opt['mean']
        self.std = opt['std']

    def __getitem__(self, index):
        img_name = self.pos_img_list_each[index]
        img_path = os.path.join(self.all_gt_folder, img_name)
        # print(img_path)

        img_gt = imfromfile(img_path, float32=True)

        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        return {'gt': img_gt, 'gt_path': img_path}

    def __len__(self):
        return len(self.pos_img_list_each)

