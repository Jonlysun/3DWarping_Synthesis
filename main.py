import cv2
import argparse
import numpy as np
import math
import os.path as osp
import os
import yaml
from DisparityView import Disp2VS
from DepthView import Depth2VS


config = yaml.load(open('argument.yml', 'r'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='input a int (from 0)')
    parser.add_argument('--f', type=int, help='frame number', default=50)
    parser.add_argument('--fill', action='store_true',
                        default=False, help='fill the hole or not')
    args = parser.parse_args()
    pid = args.id
    frame = args.f
    fill = args.fill
    print("picture id:", pid)
    print("frame number:", frame)
    print("fill:", fill)

    L_img_path = 'image/{}_l.jpg'.format(pid)
    R_img_path = 'image/{}_r.jpg'.format(pid)

    L_img = cv2.imread(L_img_path)
    R_img = cv2.imread(R_img_path)

    '''
    seman_mask = cv2.imread("2_l_simple_mask.png")[:, :, 0]
    Disp2VS(L_img, R_img, pid, frame, fill, seman_mask)
    '''
    Depth2VS(L_img, R_img, pid, frame, fill)
