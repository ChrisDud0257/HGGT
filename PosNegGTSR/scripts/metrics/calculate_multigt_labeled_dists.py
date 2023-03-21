import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import torch
import os
from collections import Counter
import argparse
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
from DISTS_pytorch import DISTS
from PIL import Image

from basicsr.utils import img2tensor
import json

def prepare_image(image, resize=False):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

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

def cropborder(imgs, border_size = 0):
    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [i[:, :, border_size:-border_size, border_size:-border_size] for i in imgs]
    if len(imgs) == 0:
        return imgs[0]
    else:
        return imgs

def main(args):
    save_txt_path = os.path.join(os.path.dirname(args.restored), 'dists_Test-100.txt')
    save_txt = open(save_txt_path, mode='w', encoding='utf-8')
    device = torch.device("cuda:1")

    dataset = os.path.basename(args.restored)
    save_txt.write(f"\nDISTS for testing dataset: {dataset}.\n")
    print(f"DISTS for testing dataset: {dataset}")

    dists_P_avg_all_list = []

    dists_model = DISTS().to(device)

    img_sr_paths = args.restored

    img_gt_paths_01 = os.path.join(args.gts, '01')
    img_gt_paths_02 = os.path.join(args.gts, '02')
    img_gt_paths_03 = os.path.join(args.gts, '03')
    img_gt_paths_04 = os.path.join(args.gts, '04')

    Ajson_path = os.path.join(args.json_path, 'A')
    Bjson_path = os.path.join(args.json_path, 'B')
    Cjson_path = os.path.join(args.json_path, 'C')

    for img_sr_name in sorted(os.listdir(img_sr_paths)):
        img_gt_name_01 = '_'.join(img_sr_name.split('_')[:-1]) + '_01.png'
        img_gt_path_01 = os.path.join(img_gt_paths_01, img_gt_name_01)
        img_gt_01 = prepare_image(Image.open(img_gt_path_01).convert("RGB"))

        img_gt_name_02 = '_'.join(img_sr_name.split('_')[:-1]) + '_02.png'
        img_gt_path_02 = os.path.join(img_gt_paths_02, img_gt_name_02)
        img_gt_02 = prepare_image(Image.open(img_gt_path_02).convert("RGB"))

        img_gt_name_03 = '_'.join(img_sr_name.split('_')[:-1]) + '_03.png'
        img_gt_path_03 = os.path.join(img_gt_paths_03, img_gt_name_03)
        img_gt_03 = prepare_image(Image.open(img_gt_path_03).convert("RGB"))

        img_gt_name_04 = '_'.join(img_sr_name.split('_')[:-1]) + '_04.png'
        img_gt_path_04 = os.path.join(img_gt_paths_04, img_gt_name_04)
        img_gt_04 = prepare_image(Image.open(img_gt_path_04).convert("RGB"))

        img_sr_path = os.path.join(img_sr_paths, img_sr_name)
        img_sr = prepare_image(Image.open(img_sr_path).convert("RGB"))

        if args.crop_border != 0:
            img_gt_01, img_gt_02, img_gt_03, img_gt_04, img_sr = cropborder(
                [img_gt_01, img_gt_02, img_gt_03, img_gt_04, img_sr], border_size=args.crop_border)

        img_gt_01 = img_gt_01.to(device)
        img_gt_02 = img_gt_02.to(device)
        img_gt_03 = img_gt_03.to(device)
        img_gt_04 = img_gt_04.to(device)
        img_sr = img_sr.to(device)

        img_gt_list = [img_gt_01, img_gt_02, img_gt_03, img_gt_04]

        Ajson_full_path = os.path.join(Ajson_path, '_'.join(img_sr_name.split('_')[:-1]) + '.json')
        Bjson_full_path = os.path.join(Bjson_path, '_'.join(img_sr_name.split('_')[:-1]) + '.json')
        Cjson_full_path = os.path.join(Cjson_path, '_'.join(img_sr_name.split('_')[:-1]) + '.json')

        with open(Ajson_full_path, mode='r', encoding='utf-8') as fA:
            jsonA = json.load(fA)
        with open(Bjson_full_path, mode='r', encoding='utf-8') as fB:
            jsonB = json.load(fB)
        with open(Cjson_full_path, mode='r', encoding='utf-8') as fC:
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

        save_txt.write(f"{'_'.join(img_sr_name.split('_')[:-1])} 01:{img1_effective_label} 02:{img2_effective_label} 03:{img3_effective_label} 04:{img4_effective_label}\n")

        label_list = [img1_effective_label, img2_effective_label, img3_effective_label, img4_effective_label]

        dists_P_list = []

        for idx, label in enumerate(label_list):

            if label == 'Positive':
                dists_P = (dists_model(img_gt_list[idx], img_sr)).item()
                dists_P_list.append(dists_P)
                save_txt.write(f"{(idx + 1):02d} DISTS:{dists_P:.4f}\n")
        dists_P_avg = sum(dists_P_list) / len(dists_P_list)
        save_txt.write(f"DISTS-avg:{dists_P_avg:.4f}\n")

        dists_P_avg_all_list.append(dists_P_avg)

    dists_P_avg_all = sum(dists_P_avg_all_list) / len(dists_P_avg_all_list)
    print(f"DISTS-avg all:{dists_P_avg_all:.4f}\n")
    save_txt.write(f"DISTS-avg all:{dists_P_avg_all:.4f}\n")

    save_txt.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gts', type=str, default='../../datasets/Test/GT/Test-100/images', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str,
                        default='../../results/BSRGAN_DF2K_OST_Blind_x4/visulization/Test-100', help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    parser.add_argument('--json_path', type=str, default='../../datasets/Test/GT/Test-100/labels')
    args = parser.parse_args()
    main(args)