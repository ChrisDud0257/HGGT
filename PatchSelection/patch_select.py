import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import math

def compute_var(img, args):
    cont_var_thresh = args.cont_var_thresh
    freq_var_thresh = args.freq_var_thresh
    if img.shape[2] == 3:
        img = Image.fromarray(img.astype(np.uint8))
        im_gray = img.convert("L")
        im_gray = np.array(im_gray)
        # im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        im_gray = img
    [_, var] = cv2.meanStdDev(im_gray.astype(np.float32))
    freq_var = cv2.Laplacian(im_gray, cv2.CV_8U).var()
    print(f"cont_var is {var[0][0]}, freq_var is {freq_var}.")
    if var[0][0] >= cont_var_thresh and freq_var >= freq_var_thresh:
        return True
    else:
        return False


def main(args):
    img_o_path = os.path.join(args.img_path, "original")
    img_1_path = os.path.join(args.img_path, "01")
    img_2_path = os.path.join(args.img_path, "02")
    img_3_path = os.path.join(args.img_path, "03")
    img_4_path = os.path.join(args.img_path, "04")

    save_o_path = os.path.join(args.save_path, "original")
    save_1_path = os.path.join(args.save_path, "01")
    save_2_path = os.path.join(args.save_path, "02")
    save_3_path = os.path.join(args.save_path, "03")
    save_4_path = os.path.join(args.save_path, "04")

    save_txt_path = os.path.join(args.save_path, "crop_area_info.txt")

    os.makedirs(save_o_path, exist_ok=True)
    os.makedirs(save_1_path, exist_ok=True)
    os.makedirs(save_2_path, exist_ok=True)
    os.makedirs(save_3_path, exist_ok=True)
    os.makedirs(save_4_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)

    txt = open(save_txt_path, 'w')
    index = 0
    for filename in os.listdir(img_o_path):
        img_name, _ = os.path.splitext(filename)
        img_o_fullpath = os.path.join(img_o_path, filename)
        img_1_fullpath = os.path.join(img_1_path, f"{img_name}_01.png")
        img_2_fullpath = os.path.join(img_2_path, f"{img_name}_02.png")
        img_3_fullpath = os.path.join(img_3_path, f"{img_name}_03.png")
        img_4_fullpath = os.path.join(img_4_path, f"{img_name}_04.png")

        img_o = Image.open(img_o_fullpath).convert("RGB")
        img_1 = Image.open(img_1_fullpath).convert("RGB")
        img_2 = Image.open(img_2_fullpath).convert("RGB")
        img_3 = Image.open(img_3_fullpath).convert("RGB")
        img_4 = Image.open(img_4_fullpath).convert("RGB")

        img_o_nd = np.array(img_o).astype(np.float32)
        img_1_nd = np.array(img_1).astype(np.float32)
        img_2_nd = np.array(img_2).astype(np.float32)
        img_3_nd = np.array(img_3).astype(np.float32)
        img_4_nd = np.array(img_4).astype(np.float32)

        h, w, _ = img_o_nd.shape

        if h >= 2400:
            k = 128
        elif h >= 1200:
            k = 64
        else:
            k = 32
        init_h = random.randint(0, k)
        h_space = np.arange(init_h, h, args.step_size)

        if w >= 2400:
            k = 128
        elif w >= 1200:
            k = 64
        else:
            k = 32
        init_w = random.randint(0, k)
        w_space = np.arange(init_w, w, args.step_size)

        max_dist_list = []
        h_w_list = []
        for x in h_space:
            for y in w_space:
                db_x = random.randint(-16, 16)
                db_y = random.randint(-16, 16)
                x_ = max(x + db_x, 0)
                y_ = max(y + db_y, 0)

                if x_ + args.crop_size <= h and y_ + args.crop_size <= w:
                    img_o_crop = img_o_nd[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
                    img_o_crop = np.ascontiguousarray(img_o_crop)
                    img_1_crop = img_1_nd[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
                    img_1_crop = np.ascontiguousarray(img_1_crop)
                    img_2_crop = img_2_nd[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
                    img_2_crop = np.ascontiguousarray(img_2_crop)
                    img_3_crop = img_3_nd[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
                    img_3_crop = np.ascontiguousarray(img_3_crop)
                    img_4_crop = img_4_nd[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
                    img_4_crop = np.ascontiguousarray(img_4_crop)

                    bg_o = compute_var(img_o_crop, args)
                    bg_1 = compute_var(img_1_crop, args)
                    bg_2 = compute_var(img_2_crop, args)
                    bg_3 = compute_var(img_3_crop, args)
                    bg_4 = compute_var(img_4_crop, args)

                    if bg_o or bg_1 or bg_2 or bg_3 or bg_4:
                        h_w = [x_, y_]
                        dist1 = abs((img_o_crop - img_1_crop).mean())
                        dist2 = abs((img_o_crop - img_2_crop).mean())
                        dist3 = abs((img_o_crop - img_3_crop).mean())
                        dist4 = abs((img_o_crop - img_4_crop).mean())
                        max_dist = max(dist1, dist2, dist3, dist4)
                        max_dist_list.append(max_dist)
                        h_w_list.append(h_w)
                        print("It doesn't have too much blank area.")
                    else:
                        print("It has too much blank area.")
                        continue
                else:
                    print("The crop area exceeds the original image.")
                    continue

        max_dist_list_sort = sorted(((v, i) for i, v in enumerate(max_dist_list)), reverse=True)

        max_dist_list_chs = max_dist_list_sort

        for i in max_dist_list_chs:
            print(i)
            index = index + 1
            x_, y_ = h_w_list[i[1]]
            print(f"x_ y_ is [{x_, y_}]")
            max_dist = i[0]
            img_o_crop = img_o_nd[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
            img_o_crop = np.ascontiguousarray(img_o_crop).astype(np.uint8)
            img_1_crop = img_1_nd[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
            img_1_crop = np.ascontiguousarray(img_1_crop).astype(np.uint8)
            img_2_crop = img_2_nd[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
            img_2_crop = np.ascontiguousarray(img_2_crop).astype(np.uint8)
            img_3_crop = img_3_nd[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
            img_3_crop = np.ascontiguousarray(img_3_crop).astype(np.uint8)
            img_4_crop = img_4_nd[x_:x_ + args.crop_size, y_:y_ + args.crop_size, :]
            img_4_crop = np.ascontiguousarray(img_4_crop).astype(np.uint8)

            img_o_crop = Image.fromarray(img_o_crop)
            img_1_crop = Image.fromarray(img_1_crop)
            img_2_crop = Image.fromarray(img_2_crop)
            img_3_crop = Image.fromarray(img_3_crop)
            img_4_crop = Image.fromarray(img_4_crop)

            img_o_crop.save(os.path.join(save_o_path, f"{img_name}-{x_}-{y_}.png"), "PNG", quality = 100)
            img_1_crop.save(os.path.join(save_1_path, f"{img_name}-{x_}-{y_}_01.png"), "PNG", quality=100)
            img_2_crop.save(os.path.join(save_2_path, f"{img_name}-{x_}-{y_}_02.png"), "PNG", quality=100)
            img_3_crop.save(os.path.join(save_3_path, f"{img_name}-{x_}-{y_}_03.png"), "PNG", quality=100)
            img_4_crop.save(os.path.join(save_4_path, f"{img_name}-{x_}-{y_}_04.png"), "PNG", quality=100)
            txt.write(f"{img_name}-{x_}-{y_}-{max_dist:.4f}-{index}.png\n")
            print(f"{img_name}-{x_}-{y_}.png has been saved.")
        print(f"The current image is {img_name}.png")
    txt.close()
    print("The overall of the saved crop area numbers are", index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type = str, default=r"J:\PosNegGTs")
    parser.add_argument("--crop_size", type = int, default=512)
    parser.add_argument("--save_path", type = str, default=r"J:\PatchSelection")
    parser.add_argument("--step_size", type = int, default=288)
    parser.add_argument("--cont_var_thresh", type = int, default=30)
    parser.add_argument("--freq_var_thresh", type = int, default=30)
    args = parser.parse_args()
    main(args)
