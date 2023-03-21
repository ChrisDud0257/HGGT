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

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, imfromfile
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class TwoStageDegradation_DF2K_Dataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(TwoStageDegradation_DF2K_Dataset, self).__init__()
        self.opt = opt
        self.all_gt_folder = opt['dataroot_all_gt']
        self.img_name_list = [i for i in os.listdir(self.all_gt_folder)]

        # blur settings for the first degradation
        self.blur_kernel_size_min = opt['blur_kernel_size_min']
        self.blur_kernel_size_max = opt['blur_kernel_size_max']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size_min2 = opt['blur_kernel_size_min2']
        self.blur_kernel_size_max2 = opt['blur_kernel_size_max2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']  # a list for each kernel probability
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']  # betag used in generalized Gaussian blur kernels
        self.betap_range2 = opt['betap_range2']  # betap used in plateau blur kernels
        self.sinc_prob2 = opt['sinc_prob2']  # the probability for sinc filters

        self.kernel_range = [2 * v + 1 for v in range(self.blur_kernel_size_min, self.blur_kernel_size_max + 1)]  # kernel size ranges from 3 to 9
        self.kernel_range2 = [2 * v + 1 for v in range(self.blur_kernel_size_min2,
                                                      self.blur_kernel_size_max2 + 1)]  # kernel size ranges from 3 to 9

        self.final_sinc_prob = opt['final_sinc_prob']


        self.pulse_tensor = torch.zeros(9, 9).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[4, 4] = 1

    def __getitem__(self, index):

        img_name = self.img_name_list[index]

        img_gt = imfromfile(os.path.join(self.all_gt_folder, img_name), float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

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
        pad_size = (9 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (9 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=9)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)


        return_d = {'gt': img_gt,
                    'kernel1': kernel,'kernel2': kernel2,
                    'sinc_kernel': sinc_kernel}
        return return_d

    def __len__(self):
        return len(self.img_name_list)
