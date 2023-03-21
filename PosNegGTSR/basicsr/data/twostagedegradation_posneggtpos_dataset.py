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
    most_num = most_counterNum[0][1]
    return most_element, most_num


@DATASET_REGISTRY.register()
class TwoStageDegradation_PosNegGTPos_Dataset(data.Dataset):
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
        super(TwoStageDegradation_PosNegGTPos_Dataset, self).__init__()
        self.opt = opt
        self.all_gt_folder = opt['dataroot_all_gt']  # 0 - 20000 sets of images folder. The folder has five sub folders, i.e. original, 01, 02, 03, 04, 05, each sub folder contains different GTs.
        self.all_json_folder = opt['dataroot_all_json']  # 0 - 20000 labels, A, B, C sub folders, represents one set of images has three different labels from three different people.
        # Filter out the sets which don't have any effective P labels at all.

        self.all_json = [json_file for json_file in os.listdir(os.path.join(self.all_json_folder, 'A'))]
        self.no_effective_set = self.filter_all_no_positive_set(self.all_json_folder)
        # Get the final effective sets, the four images of which has at least have one effective P label
        self.effective_json = [json_file for json_file in self.all_json if json_file not in self.no_effective_set]

        self.effective_img = []
        for json_file in self.effective_json:
            img_name = os.path.splitext(json_file)[0]
            self.effective_img.append(img_name)

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

        json_file = self.effective_json[index]
        img_name = self.effective_img[index]

        img_original_path = osp.join(self.all_gt_folder, "original", f"{img_name}.png")
        img_01_path = osp.join(self.all_gt_folder, "01", f"{img_name}_01.png")
        img_02_path = osp.join(self.all_gt_folder, "02", f"{img_name}_02.png")
        img_03_path = osp.join(self.all_gt_folder, "03", f"{img_name}_03.png")
        img_04_path = osp.join(self.all_gt_folder, "04", f"{img_name}_04.png")

        img_o = imfromfile(img_original_path, float32=True)
        img_01 = imfromfile(img_01_path, float32=True)
        img_02 = imfromfile(img_02_path, float32=True)
        img_03 = imfromfile(img_03_path, float32=True)
        img_04 = imfromfile(img_04_path, float32=True)

        img_list = [img_o, img_01, img_02, img_03, img_04]

        self.Ajson_path = os.path.join(self.all_json_folder, 'A', json_file)
        self.Bjson_path = os.path.join(self.all_json_folder, 'B', json_file)
        self.Cjson_path = os.path.join(self.all_json_folder, 'C', json_file)

        with open(self.Ajson_path, mode='r', encoding='utf-8') as fA:
            jsonA = json.load(fA)
        with open(self.Bjson_path, mode='r', encoding='utf-8') as fB:
            jsonB = json.load(fB)
        with open(self.Cjson_path, mode='r', encoding='utf-8') as fC:
            jsonC = json.load(fC)

        Alabel_img1 = jsonA['Picture_2']['Label']
        Alabel_img2 = jsonA['Picture_3']['Label']
        Alabel_img3 = jsonA['Picture_4']['Label']
        Alabel_img4 = jsonA['Picture_5']['Label']

        Blabel_img1 = jsonB['Picture_2']['Label']
        Blabel_img2 = jsonB['Picture_3']['Label']
        Blabel_img3 = jsonB['Picture_4']['Label']
        Blabel_img4 = jsonB['Picture_5']['Label']

        Clabel_img1 = jsonC['Picture_2']['Label']
        Clabel_img2 = jsonC['Picture_3']['Label']
        Clabel_img3 = jsonC['Picture_4']['Label']
        Clabel_img4 = jsonC['Picture_5']['Label']

        img1_label = [Alabel_img1, Blabel_img1, Clabel_img1]
        img2_label = [Alabel_img2, Blabel_img2, Clabel_img2]
        img3_label = [Alabel_img3, Blabel_img3, Clabel_img3]
        img4_label = [Alabel_img4, Blabel_img4, Clabel_img4]

        imgo_label = 'Original'
        imgo_label_num = 3

        # Give final label to every image in a set, the final label is also called the effective label.
        if len(img1_label) == len(set(img1_label)):
            img1_effective_label = 'Similar'
            img1_label_num = 1
        else:
            img1_effective_label, img1_label_num = find_max_repetition(*img1_label)

        if len(img2_label) == len(set(img2_label)):
            img2_effective_label = 'Similar'
            img2_label_num = 1
        else:
            img2_effective_label, img2_label_num = find_max_repetition(*img2_label)

        if len(img3_label) == len(set(img3_label)):
            img3_effective_label = 'Similar'
            img3_label_num = 1
        else:
            img3_effective_label, img3_label_num = find_max_repetition(*img3_label)

        if len(img4_label) == len(set(img4_label)):
            img4_effective_label = 'Similar'
            img4_label_num = 1
        else:
            img4_effective_label, img4_label_num = find_max_repetition(*img4_label)

        label_list = [imgo_label, img1_effective_label, img2_effective_label, img3_effective_label,
                      img4_effective_label]
        label_list_array = []

        label_num_list = [imgo_label_num, img1_label_num, img2_label_num, img3_label_num, img4_label_num]
        label_num_list = np.stack(label_num_list, axis=0)

        # Since label string could not be converted into torch.tensor or np.array, then we firstly convert the labels into the formal of int
        for str in label_list:
            if str == "Original":
                k = 0
            elif str == "Positive":
                k = 1
            elif str == "Similar":
                k = 2
            elif str == "Negative":
                k = 3
            else:
                print(f"wrong label value {str}.")
            label_list_array.append(k)
        # print(f"label_list is {label_list}")
        label_list_array = np.stack(label_list_array, axis=0)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_list = augment(img_list, self.opt['use_hflip'], self.opt['use_rot'])

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
        img_list = img2tensor(img_list, bgr2rgb=True, float32=True)
        img_list = torch.stack(img_list, dim=0)

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'img_list': img_list,
                    'label_list': label_list_array,
                    'label_num_list': label_num_list,
                    'kernel1': kernel,
                    'kernel2':kernel2,
                    'sinc_kernel': sinc_kernel}
        return return_d

    def __len__(self):
        return len(self.effective_json)

    def filter_all_no_positive_set(self, all_json_folder):
        Apath = os.path.join(all_json_folder, 'A')
        Bpath = os.path.join(all_json_folder, 'B')
        Cpath = os.path.join(all_json_folder, 'C')

        no_effective_set = []
        for json_file in os.listdir(Apath):
            Ajson_path = os.path.join(Apath, json_file)
            Bjson_path = os.path.join(Bpath, json_file)
            Cjson_path = os.path.join(Cpath, json_file)

            with open(Ajson_path, mode='r', encoding='utf-8') as fA:
                jsonA = json.load(fA)
            with open(Bjson_path, mode='r', encoding='utf-8') as fB:
                jsonB = json.load(fB)
            with open(Cjson_path, mode='r', encoding='utf-8') as fC:
                jsonC = json.load(fC)

            # Each set has four enhanced GTs
            Alabel_img1 = jsonA['Picture_2']['Label']
            Alabel_img2 = jsonA['Picture_3']['Label']
            Alabel_img3 = jsonA['Picture_4']['Label']
            Alabel_img4 = jsonA['Picture_5']['Label']

            Blabel_img1 = jsonB['Picture_2']['Label']
            Blabel_img2 = jsonB['Picture_3']['Label']
            Blabel_img3 = jsonB['Picture_4']['Label']
            Blabel_img4 = jsonB['Picture_5']['Label']

            Clabel_img1 = jsonC['Picture_2']['Label']
            Clabel_img2 = jsonC['Picture_3']['Label']
            Clabel_img3 = jsonC['Picture_4']['Label']
            Clabel_img4 = jsonC['Picture_5']['Label']

            # One image has three labels from three different people
            img1_label = [Alabel_img1, Blabel_img1, Clabel_img1]
            img2_label = [Alabel_img2, Blabel_img2, Clabel_img2]
            img3_label = [Alabel_img3, Blabel_img3, Clabel_img3]
            img4_label = [Alabel_img4, Blabel_img4, Clabel_img4]

            img1_p_num, _, _ = check_list(*img1_label)
            img2_p_num, _, _ = check_list(*img2_label)
            img3_p_num, _, _ = check_list(*img3_label)
            img4_p_num, _, _ = check_list(*img4_label)

            # p_num <= 1 means the final label of the image wouldn't be P, since the P label numbers are not more than 1, when three different people give labels to the same image.
            if img1_p_num <= 1 and img2_p_num <= 1 and img3_p_num <= 1 and img4_p_num <= 1:
                no_effective_set.append(json_file)

        return  no_effective_set
