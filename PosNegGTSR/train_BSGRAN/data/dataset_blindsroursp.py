import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
from utils import utils_blindsr as blindsr

import json
from collections import Counter

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

class DatasetBlindSROursP(data.Dataset):
    def __init__(self, opt):
        super(DatasetBlindSROursP, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.shuffle_prob = opt['shuffle_prob'] if opt['shuffle_prob'] else 0.1
        self.use_sharp = opt['use_sharp'] if opt['use_sharp'] else False
        self.degradation_type = opt['degradation_type'] if opt['degradation_type'] else 'bsrgan'
        self.lq_patchsize = self.opt['lq_patchsize'] if self.opt['lq_patchsize'] else 64
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else self.lq_patchsize * self.sf

        self.all_gt_folder = self.opt['dataroot_all_gt']  # 0 - 20000 sets of images folder. The folder has five sub folders, i.e. original, 01, 02, 03, 04, 05, each sub folder contains different GTs.
        self.all_json_folder = self.opt['dataroot_all_json']

        # Filter out the sets which don't have any effective P labels at all.
        self.no_effective_set = self.filter_no_effective_set(self.all_json_folder)

        self.all_json = [json_file for json_file in os.listdir(os.path.join(self.all_json_folder, 'A'))]
        # Get the final effective sets, the four images of which has at least have one effective P label
        self.effective_json = [json_file for json_file in self.all_json if json_file not in self.no_effective_set]

        self.effective_img = []
        for json_file in self.effective_json:
            img_name = os.path.splitext(json_file)[0]
            self.effective_img.append(img_name)

        # self.paths_H = util.get_image_paths(opt['dataroot_H'])
        # print(len(self.paths_H))

        #        for n, v in enumerate(self.paths_H):
        #            if 'face' in v:
        #                del self.paths_H[n]
        #        time.sleep(1)
        # assert self.paths_H, 'Error: H path is empty.'

    def __getitem__(self, index):

        L_path = None

        json_file = self.effective_json[index]
        img_name = self.effective_img[index]

        img_original_path = os.path.join(self.all_gt_folder, "original", f"{img_name}.png")
        img_01_path = os.path.join(self.all_gt_folder, "01", f"{img_name}_01.png")
        img_02_path = os.path.join(self.all_gt_folder, "02", f"{img_name}_02.png")
        img_03_path = os.path.join(self.all_gt_folder, "03", f"{img_name}_03.png")
        img_04_path = os.path.join(self.all_gt_folder, "04", f"{img_name}_04.png")

        img_list_path = [img_original_path, img_01_path, img_02_path, img_03_path, img_04_path]

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

        label_list = [imgo_label, img1_effective_label, img2_effective_label, img3_effective_label, img4_effective_label]
        label_list_array = []

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
        l = [item == 1 for item in label_list_array]
        index = np.where(np.array(l) == True)[0]
        idx = random.choice(index)

        img_random_H_enhance_path = img_list_path[idx]

        # ------------------------------------
        # get H image
        # ------------------------------------

        img_H = util.imread_uint(img_original_path, self.n_channels)
        img_name, ext = os.path.splitext(os.path.basename(img_original_path))
        H, W, C = img_H.shape

        img_H_enhance = util.imread_uint(img_random_H_enhance_path, self.n_channels)

        # if H < self.patch_size or W < self.patch_size:
        #     img_H = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8),
        #                     (self.patch_size, self.patch_size, 1))
        #     img_H_enhance = np.tile(np.random.randint(0, 256, size=[1, 1, self.n_channels], dtype=np.uint8),
        #                     (self.patch_size, self.patch_size, 1))

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_H.shape

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            img_H_enhance = img_H_enhance[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            if 'face' in img_name:
                mode = random.choice([0, 4])
                img_H = util.augment_img(img_H, mode=mode)
                img_H_enhance = util.augment_img(img_H_enhance, mode=mode)
            else:
                mode = random.randint(0, 7)
                img_H = util.augment_img(img_H, mode=mode)
                img_H_enhance = util.augment_img(img_H_enhance, mode=mode)

            img_H = util.uint2single(img_H)
            img_H_enhance = util.uint2single(img_H_enhance)
            if self.degradation_type == 'bsrgan':
                img_L, img_H, img_H_enhance = blindsr.degradation_bsrgan_ours_p(img_H, img_H_enhance, self.sf, lq_patchsize=self.lq_patchsize,
                                                          isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob,
                                                               use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        else:
            img_H = util.uint2single(img_H)
            if self.degradation_type == 'bsrgan':
                img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, lq_patchsize=self.lq_patchsize,
                                                          isp_model=None)
            elif self.degradation_type == 'bsrgan_plus':
                img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, self.sf, shuffle_prob=self.shuffle_prob,
                                                               use_sharp=self.use_sharp, lq_patchsize=self.lq_patchsize)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        img_H_enhance = util.single2tensor3(img_H_enhance)

        if L_path is None:
            L_path = img_original_path

        return {'L': img_L, 'H': img_H_enhance, 'L_path': L_path, 'H_path': img_random_H_enhance_path}

    def __len__(self):
        return len(self.effective_json)

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