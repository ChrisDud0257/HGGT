import os
from PIL import Image
import cv2
import numpy as np
import argparse
import random
import math

def main(args):
    img_o_path = os.path.join(args.img_path, "original")
    img_1_path = os.path.join(args.img_path, "01")
    img_2_path = os.path.join(args.img_path, "02")
    img_3_path = os.path.join(args.img_path, "03")
    img_4_path = os.path.join(args.img_path, "04")

    listdir = os.listdir(img_o_path)
    all_img_nums = len(listdir)

    num_subdirs = math.ceil(all_img_nums / args.num_imgs)

    for _ in range(args.shuffle_nums):
        random.shuffle(listdir)

    for i in range(1,num_subdirs + 1):
        save_o_path = os.path.join(args.save_path, f"{i:02d}_subdir", f"images{(i-1)*args.num_imgs}-{i*args.num_imgs}", "original")
        save_1_path = os.path.join(args.save_path, f"{i:02d}_subdir", f"images{(i-1)*args.num_imgs}-{i*args.num_imgs}", "01")
        save_2_path = os.path.join(args.save_path, f"{i:02d}_subdir", f"images{(i-1)*args.num_imgs}-{i*args.num_imgs}", "02")
        save_3_path = os.path.join(args.save_path, f"{i:02d}_subdir", f"images{(i-1)*args.num_imgs}-{i*args.num_imgs}", "03")
        save_4_path = os.path.join(args.save_path, f"{i:02d}_subdir", f"images{(i-1)*args.num_imgs}-{i*args.num_imgs}", "04")

        tagging_label_path = os.path.join(args.save_path, f"{i:02d}_subdir",
                                           f"labels{(i - 1) * args.num_imgs}-{i * args.num_imgs}", "tagging")

        save_txt_path = os.path.join(args.save_path, f"{i:02d}_subdir", f"img_info_{(i-1)*args.num_imgs}-{i*args.num_imgs}.txt")

        os.makedirs(save_o_path, exist_ok=True)
        os.makedirs(save_1_path, exist_ok=True)
        os.makedirs(save_2_path, exist_ok=True)
        os.makedirs(save_3_path, exist_ok=True)
        os.makedirs(save_4_path, exist_ok=True)

        os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
        os.makedirs(tagging_label_path, exist_ok=True)

        start = (i-1) * args.num_imgs
        if i * args.num_imgs <= all_img_nums:
            end = i * args.num_imgs
        else:
            end = all_img_nums

        txt = open(save_txt_path, 'w')

        for filename in sorted(listdir[start:end]):
            img_name, _ = os.path.splitext(filename)
            img_o_fullpath = os.path.join(img_o_path, filename)
            img_1_fullpath = os.path.join(img_1_path, f"{img_name}_01.png")
            img_2_fullpath = os.path.join(img_2_path, f"{img_name}_02.png")
            img_3_fullpath = os.path.join(img_3_path, f"{img_name}_03.png")
            img_4_fullpath = os.path.join(img_4_path, f"{img_name}_04.png")

            os.system(f"copy {img_o_fullpath} {save_o_path}")
            os.system(f"copy {img_1_fullpath} {save_1_path}")
            os.system(f"copy {img_2_fullpath} {save_2_path}")
            os.system(f"copy {img_3_fullpath} {save_3_path}")
            os.system(f"copy {img_4_fullpath} {save_4_path}")

            txt.write(f"{filename}\n")

            print(f"{filename} has been copied.")
        txt.close()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--img_path", type=str, default=r"J:\PatchSelection")
    parse.add_argument("--save_path", type=str, default=r"J:\regroup")
    parse.add_argument("--num_imgs", type=int, default=1000)
    parse.add_argument("--shuffle_nums", type=int, default=3)
    args=parse.parse_args()
    main(args)