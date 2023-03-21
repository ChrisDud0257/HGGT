import argparse
import cv2
import numpy as np
from os import path as osp
from collections import Counter

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import bgr2ycbcr, scandir
import os
import json

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

def main(args):
    save_txt_path = os.path.join(os.path.dirname(args.restored), 'psnr_ssim_Test-100.txt')
    save_txt = open(save_txt_path, mode='w', encoding='utf-8')

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    dataset = os.path.basename(args.restored)
    save_txt.write(f"\nPSNR/SSIM for testing dataset: {dataset}.\n")
    print(f"PSNR/SSIM for testing dataset: {dataset}")

    psnr_P_avg_all_list = []
    ssim_P_avg_all_list = []

    img_sr_paths = args.restored

    img_gt_paths_01 = os.path.join(args.gts, '01')
    img_gt_paths_02 = os.path.join(args.gts, '02')
    img_gt_paths_03 = os.path.join(args.gts, '03')
    img_gt_paths_04 = os.path.join(args.gts, '04')

    Ajson_path = os.path.join(args.json_path, 'A')
    Bjson_path = os.path.join(args.json_path, 'B')
    Cjson_path = os.path.join(args.json_path, 'C')

    for img_sr_name in sorted(os.listdir(img_sr_paths)):
        img_sr_path = os.path.join(img_sr_paths, img_sr_name)
        img_sr = cv2.imread(img_sr_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_gt_name_01 = '_'.join(img_sr_name.split('_')[:-1]) + '_01.png'
        img_gt_path_01 = os.path.join(img_gt_paths_01, img_gt_name_01)
        img_gt_01 = cv2.imread(img_gt_path_01, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_gt_name_02 = '_'.join(img_sr_name.split('_')[:-1]) + '_02.png'
        img_gt_path_02 = os.path.join(img_gt_paths_02, img_gt_name_02)
        img_gt_02 = cv2.imread(img_gt_path_02, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_gt_name_03 = '_'.join(img_sr_name.split('_')[:-1]) + '_03.png'
        img_gt_path_03 = os.path.join(img_gt_paths_03, img_gt_name_03)
        img_gt_03 = cv2.imread(img_gt_path_03, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_gt_name_04 = '_'.join(img_sr_name.split('_')[:-1]) + '_04.png'
        img_gt_path_04 = os.path.join(img_gt_paths_04, img_gt_name_04)
        img_gt_04 = cv2.imread(img_gt_path_04, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if args.test_y_channel and img_gt_01.ndim == 3 and img_gt_01.shape[2] == 3:
            img_gt_01 = bgr2ycbcr(img_gt_01, y_only=True)
            img_gt_02 = bgr2ycbcr(img_gt_02, y_only=True)
            img_gt_03 = bgr2ycbcr(img_gt_03, y_only=True)
            img_gt_04 = bgr2ycbcr(img_gt_04, y_only=True)

            img_sr = bgr2ycbcr(img_sr, y_only=True)

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

        p_num, _, _ = check_list(*label_list)
        if p_num < 2:
            print(f"{'_'.join(img_sr_name.split('_')[:-1])} has less than 2 positive GTs.")

        psnr_P_list = []
        ssim_P_list = []

        for idx, label in enumerate(label_list):
            if label == 'Positive':
                psnr_P = calculate_psnr(img_gt_list[idx] * 255, img_sr * 255, crop_border=args.crop_border, input_order='HWC')
                ssim_P = calculate_ssim(img_gt_list[idx] * 255, img_sr * 255, crop_border=args.crop_border, input_order='HWC')
                psnr_P_list.append(psnr_P)
                ssim_P_list.append(ssim_P)
                save_txt.write(f"{(idx+1):02d} PSNR:{psnr_P:.4f} SSIM:{ssim_P:.4f}\n")
        psnr_P_avg = sum(psnr_P_list) / len(psnr_P_list)
        ssim_P_avg = sum(ssim_P_list) / len(ssim_P_list)
        save_txt.write(f"PSNR-avg:{psnr_P_avg:.4f} SSIM-avg:{ssim_P_avg:.4f}\n")

        psnr_P_avg_all_list.append(psnr_P_avg)
        ssim_P_avg_all_list.append(ssim_P_avg)

    psnr_P_avg_all = sum(psnr_P_avg_all_list) / len(psnr_P_avg_all_list)
    ssim_P_avg_all = sum(ssim_P_avg_all_list) / len(ssim_P_avg_all_list)

    print(f"PSNR-avg all:{psnr_P_avg_all:.4f} SSIM-avg all:{ssim_P_avg_all:.4f}\n")
    save_txt.write(f"PSNR-avg all:{psnr_P_avg_all:.4f} SSIM-avg all:{ssim_P_avg_all:.4f}\n")

    save_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gts', type=str, default='../../datasets/Test/GT/Test-100/images', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str,
                        default='../../results/BSRGAN_DF2K_OST_Blind_x4/visulization/Test-100', help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='_baseline', help='Suffix for restored images')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    parser.add_argument('--json_path', type=str, default='../../datasets/Test/GT/Test-100/labels')
    args = parser.parse_args()
    main(args)