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
from basicsr.data.transforms import augment, paired_random_crop
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
class PairedLabeledImageDataset(data.Dataset):
    """
    Every image has three labels from three different people, if the three labels at least have two same labels, then the final label of the image will be attributed as
    the most common label. If the labels are all different from each other, i.e. one is P, one is S, one is N, then the final label will be S, S will not participate in
    our training progress. The final label here we call the effective label.
    In a set of images, if the four effective labels don't have any P, then we will firstly filter out these sets.
    """

    def __init__(self, opt):
        super(PairedLabeledImageDataset, self).__init__()
        self.opt = opt
        self.all_gt_folder = opt['dataroot_all_gt']  # 0 - 20000 sets of images folder. The folder has five sub folders, i.e. original, 01, 02, 03, 04, 05, each sub folder contains different GTs.
        self.all_bicubic_LR_folder = opt['dataroot_all_bicubic_LR_folder']
        self.all_json_folder = opt['dataroot_all_json'] # 0 - 20000 labels, A, B, C sub folders, represents one set of images has three different labels from three different people.
        # Filter out the sets which don't have any effective P labels at all.

        if self.opt['filter_no_effective_set']:

            self.no_effective_set = self.filter_no_effective_set(self.all_json_folder)

            self.all_json = [json_file for json_file in os.listdir(os.path.join(self.all_json_folder, 'A'))]
            # Get the final effective sets, the four images of which has at least have one effective P label
            self.effective_json = [json_file for json_file in self.all_json if json_file not in self.no_effective_set]

            self.effective_img = []
            for json_file in self.effective_json:
                img_name = os.path.splitext(json_file)[0]
                self.effective_img.append(img_name)
        else:
            self.effective_img = [os.path.splitext(json_file)[0] for json_file in os.listdir(os.path.join(self.all_json_folder, 'A'))]


    def filter_no_effective_set(self, all_json_folder):
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

    def __getitem__(self, index):
        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_name = self.effective_img[index]

        scale = self.opt['scale']

        img_original_path = osp.join(self.all_gt_folder, "original", f"{img_name}.png")
        img_o = imfromfile(img_original_path, float32=True)

        img_bicubic_LR_path = osp.join(self.all_bicubic_LR_folder, "original", f"{img_name}.png")
        img_bicubic_LR = imfromfile(img_bicubic_LR_path, float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_o, img_bicubic_LR = paired_random_crop(img_o, img_bicubic_LR, gt_size, scale)
            img_o, img_bicubic_LR = augment([img_o, img_bicubic_LR], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_o = img2tensor(img_o, bgr2rgb=True, float32=True)
        img_bicubic_LR = img2tensor(img_bicubic_LR, bgr2rgb=True, float32=True)

        return_d = {'gt': img_o,
                    'lq': img_bicubic_LR}
        return return_d

    def __len__(self):
        return len(self.effective_img)
