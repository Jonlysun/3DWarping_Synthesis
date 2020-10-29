import cv2
from PIL import Image
import numpy as np
import os.path as osp
from utils import getInstanceColorImage, make_palette, color_seg
import argparse


def getDispSeman(ins_img_mask):
    disp_img = cv2.imread("disparity/6_l.png")
    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2GRAY)
    sub_disp_img = ins_img_mask * disp_img
    cv2.imwrite("sub_disp_img.jpg", sub_disp_img)

def getSimpleMask(obj_num, instance_id):
    instance_id[instance_id != obj_num] = 0
    ins_img_mask = np.clip(instance_id, 0, 1)
    return ins_img_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int)
    args = parser.parse_args()
    pid = args.id

    sem_dir_name = "Semantic"
    img_dir_name = "image"

    pic_name = "{}_l.png".format(pid)
    img_name = "{}_l.jpg".format(pid)

    sem_path = osp.join(sem_dir_name, pic_name)
    img_path = osp.join(img_dir_name, img_name)

    sem_img = Image.open(sem_path, "r")
    sem_img = np.array(sem_img)
    semantic_id = (sem_img >> 8).astype(np.uint8)
    instance_id = sem_img.astype(np.uint8)
    instance_id += semantic_id

    color_num = np.max(semantic_id) + np.max(instance_id)
    palette = make_palette(color_num)
    ins_img = color_seg(instance_id, palette)
    cv2.imwrite("Semantic_color/{}_l.jpg".format(pid), ins_img)

    obj_num = 34
    instance_id[instance_id != obj_num] = 0
    instance_id = Image.fromarray(instance_id)
    instance_id.save("2_l_simple_mask.png")

    img = cv2.imread(img_path)
    ins_img_mask = getSimpleMask(34, np.array(instance_id))
    sub_img_mask = cv2.merge([ins_img_mask, ins_img_mask, ins_img_mask])
    sub_img = sub_img_mask * img
    cv2.imwrite("{}_l.jpg".format(pid), sub_img)
    