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
class PosNegGTOriDataset(data.Dataset):
    """
    Every image has three labels from three different people, if the three labels at least have two same labels, then the final label of the image will be attributed as
    the most common label. If the labels are all different from each other, i.e. one is P, one is S, one is N, then the final label will be S, S will not participate in
    our training progress. The final label here we call the effective label.
    In a set of images, if the four effective labels don't have any P, then we will firstly filter out these sets.
    """

    def __init__(self, opt):
        super(PosNegGTOriDataset, self).__init__()
        self.opt = opt
        self.all_gt_folder = opt['dataroot_all_gt']  # 0 - 20000 sets of images folder. The folder has five sub folders, i.e. original, 01, 02, 03, 04, 05, each sub folder contains different GTs.
        self.all_json_folder = opt['dataroot_all_json'] # 0 - 20000 labels, A, B, C sub folders, represents one set of images has three different labels from three different people.

        self.effective_img = [os.path.splitext(json_file)[0] for json_file in os.listdir(os.path.join(self.all_json_folder, 'A'))]

        # blur settings for the first degradation
        self.blur_kernel_size_min = opt['blur_kernel_size_min']
        self.blur_kernel_size_max = opt['blur_kernel_size_max']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        self.kernel_range = [2 * v + 1 for v in range(self.blur_kernel_size_min, self.blur_kernel_size_max + 1)]  # kernel size ranges from 3 to 9

    def __getitem__(self, index):
        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_name = self.effective_img[index]

        img_original_path = osp.join(self.all_gt_folder, "original", f"{img_name}.png")
        img_o = imfromfile(img_original_path, float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_o = augment(img_o, self.opt['use_hflip'], self.opt['use_rot'])

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (self.kernel_range[-1] - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_o = img2tensor(img_o, bgr2rgb=True, float32=True)

        kernel = torch.FloatTensor(kernel)

        return_d = {'gt': img_o,
                    'kernel1': kernel}
        return return_d

    def __len__(self):
        return len(self.effective_img)
