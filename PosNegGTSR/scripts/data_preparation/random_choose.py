import argparse
import math
import os
import random
import cv2

def main(args):
    all_img_path = os.listdir(args.imgpath)
    os.makedirs(args.savepath, exist_ok=True)
    for _ in range(3):
        random.shuffle(all_img_path)
    for img_dir in all_img_path[:args.number]:
        img = os.path.join(args.imgpath, img_dir)
        im = cv2.imread(img)
        h, w, c = im.shape
        if h<512 or w < 512:
            print(f"The image {img} shape is {h,w,c}.")
        os.system(f'cp -r {img} {args.savepath}')



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--imgpath", type = str, default='/home/chendu/data2_hdd10t/chendu/dataset/DF2K_OST/subimages_512')
    parse.add_argument("--savepath", type = str, default='/home/chendu/data2_hdd10t/chendu/dataset/DF2K_OST/random_choose')
    parse.add_argument("--number", type = int, default=18735)
    args = parse.parse_args()
    main(args)