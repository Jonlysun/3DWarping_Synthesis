import cv2
import numpy as np
from PIL import Image
from utils import index
import math
from utils import insertDepth


def simple_obtainNewDisp(disp, interp, semantic):
    H, W, C = disp.shape
    NewDisp = np.zeros((H, W))
    pMask = np.zeros((H, W), np.uint8)
    for i in range(H):
        for j in range(W):
            disp_value = disp[i, j, 0]
            new_disp_value = disp_value * interp
            # ???
            inew = int(j - new_disp_value)
            inew = index(inew, W)
            NewDisp[i, inew] = new_disp_value
            if semantic[i, j] != 0:
                pMask[i, inew] = 1
    disp = Image.fromarray(NewDisp.astype(np.uint8))
    NewDisp = simple_insert(NewDisp, pMask)
    disp_1 = Image.fromarray(NewDisp.astype(np.uint8))
    NewDisp = insertDepth(NewDisp)
    disp_2 = Image.fromarray(NewDisp.astype(np.uint8))
    return NewDisp.astype(np.uint8), pMask


def simple_insert(disp, pMask):
    im_floodfill = disp.copy().astype(np.uint8)
    h, w = disp.shape[:2]

    pMask = np.clip(pMask, 0, 1)

    mask = np.ones([h+2, w+2], np.uint8)
    cont_mask = np.nonzero(pMask)
    cont_mask = (cont_mask[0] + 1, cont_mask[1] + 1)
    mask[cont_mask] = 0

    disp_mask = disp * pMask
    value = int(disp_mask.sum() / pMask.sum())
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    return im_floodfill


def simple_obtainNewView(img, disp, pMask):
    H, W, C = img.shape
    NewView = np.zeros((H, W, C))
    for i in range(H):
        for j in range(W):
            if pMask[i, j] == 0:
                continue
            disp_value = disp[i, j]
            id = j + disp_value

            id0 = int(math.floor(id))
            id1 = id0 + 1
            weight1 = 1 - (id - id0)
            weight2 = id - id0
            id0 = index(id0, W)
            id1 = index(id1, W)
            NewView[i, j] = weight1 * img[i, id0] + weight2 * img[i, id1]
    return NewView


if __name__ == "__main__":
    seman_mask = cv2.imread("2_l_simple_mask.png")[:, :, 0]
    depth = cv2.imread("disparity/3_l.png")[:, :, 0]
    obj_num = np.max(seman_mask)

    '''
    disp = cv2.imread('disparity/7_l.png')
    img = cv2.imread('image/7_l.jpg')
    print(disp.shape)
    New_disp, pMask = simple_obtainNewDisp(disp, 1., seman_mask)
    NewView = simple_obtainNewView(img, New_disp, pMask)
    cv2.imwrite("simple_view.jpg", NewView)
    '''
    mask = np.clip(seman_mask, 0, 1)
    depth = depth * mask
    Image.fromarray(depth).save("test.png")
    depth[depth != 0] = np.sum(depth) / np.sum(mask)
    Image.fromarray(depth).save("test_mean.png")