import cv2
import numpy as np
from PIL import Image
import os
import imageio


def index(m, n):
    if m >= 0 and m < n:
        return m
    elif m < 0:
        return 0
    elif m >= n:
        return n-1
        
def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

    Takes:
        num_classes: the number of classes 输入为类别数目
    Gives:
        palette: the colormap as a k x 3 array of RGB colors 输出为k×3大小的RGB颜色数组
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:  #按一定规则移位产生调色板
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i)) #>>为二进制右移
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette

def color_seg(seg, palette):
    """
    Replace classes with their colors.

    Takes:
        seg: H x W segmentation image of class IDs 
    Gives:
        H x W x 3 image of class colors
    """
    return palette[seg.flat].reshape(seg.shape + (3,))

def getInstanceColorImage(sem_img):
    semantic_id = (sem_img >> 8).astype(np.uint8)
    instance_id = sem_img.astype(np.uint8)
    print(semantic_id.dtype)
    print(instance_id.dtype)
    print(np.min(semantic_id))
    print(np.min(instance_id))
    color_num = np.max(semantic_id) + np.max(instance_id)
    palette = make_palette(color_num)
    """
    img = color_seg(semantic_id, palette)
    ins_img = color_seg(instance_id, palette)
    """
    instance_id += semantic_id
    # cv2.imwrite('test.jpg', img)
    ins_img = color_seg(instance_id, palette)
    return ins_img

def read_MiDaS_depth(disp_fi, disp_rescale=10., h=None, w=None):
    if 'npy' in os.path.splitext(disp_fi)[-1]:
        disp = np.load(disp_fi)
    else:
        disp = imageio.imread(disp_fi).astype(np.float32)[:, :, 0]
    disp = disp - disp.min()
    disp = cv2.blur(disp / disp.max(), ksize=(3, 3)) * disp.max()
    disp = (disp / disp.max()) * disp_rescale
    if h is not None and w is not None:
        disp = resize(disp / disp.max(), (h, w), order=1) * disp.max()
    depth = 1. / np.maximum(disp, 0.05)

    return depth

def depth2disp(depth):
    # depth = depth - depth.min()
    # depth - (depth / depth.max()) * 10.0
    disp = 1. / depth
    return disp

def write_depth(path, depth, bits=1):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    # write_pfm(path + ".pfm", depth.astype(np.float32))

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))
        
    return

def insertDepth(img):
    Height, Width = img.shape

    new_depth = img.copy()
    # integral = img.copy()
    ptsMap = np.zeros((Height, Width))
    integral = np.zeros((Height, Width))
    # ptsMap[np.nonzero(img)] = 1

    for i in range(Height):
        for j in range(Width):
            if img[i, j] > 1e-3:
                integral[i, j] = img[i, j]
                ptsMap[i, j] = 1

    # integral
    for i in range(0, Height):
        for j in range(1, Width):
            integral[i, j] += integral[i, j-1]
            ptsMap[i, j] += ptsMap[i, j-1]
    for i in range(1, Height):
        for j in range(0, Width):
            integral[i, j] += integral[i-1, j]
            ptsMap[i, j] += ptsMap[i-1, j]

    # median filter using integral graph
    filter_size = 10
    while filter_size > 1:
        wnd = int(filter_size)
        filter_size /= 2
        for i in range(Height):
            for j in range(Width):
                left = max(0, j - wnd - 1)
                right = min(Width - 1, j + wnd)
                up = max(0, i - wnd - 1)
                bot = min(Height - 1, i + wnd)
                ptsCnt = int(ptsMap[bot, right]) + int(ptsMap[up, left]) - \
                    (int(ptsMap[up, right]) + int(ptsMap[bot, left]))
                sumGray = int(integral[bot, right]) + int(integral[up, left]) - \
                    (int(integral[up, right]) + int(integral[bot, left]))
                if ptsCnt <= 0:
                    continue
                new_depth[i, j] = float(sumGray / ptsCnt)
    # new_depth = cv2.GaussianBlur(new_depth, (3, 3), 0)
        s = int(wnd / 2) * 2 + 1
        new_depth = cv2.GaussianBlur(new_depth, (s, s), s, s)
    return new_depth