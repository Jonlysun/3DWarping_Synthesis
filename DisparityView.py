import cv2
import argparse
import numpy as np
import math
import os.path as osp
import os
from bilateral_filtering import sparse_bilateral_filtering
from utils import read_MiDaS_depth, write_depth, depth2disp, index, insertDepth
from simple import simple_obtainNewView, simple_obtainNewDisp

def obtainNewDisp(disp, interp):
    H, W, C = disp.shape
    NewDisp = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            disp_value = disp[i, j, 0]
            new_disp_value = disp_value * interp
            # ???
            inew = int(j - new_disp_value)
            inew = index(inew, W)
            NewDisp[i, inew] = new_disp_value
    return NewDisp.astype(np.uint8)


def obtainNewView(img, disp):
    H, W, C = img.shape
    NewView = np.zeros((H, W, C))
    for i in range(H):
        for j in range(W):
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


def Disp2SingleView(L_disp, interp, fill, seman_mask=None):
    NewViewDisp, pMask = simple_obtainNewDisp(L_disp, interp, seman_mask)
    if fill is True:
        NewViewDisp = insertDepth(NewViewDisp).astype(np.uint8)
    disp_img = np.zeros_like(L_disp)
    disp_img[:, :, 0] = NewViewDisp
    disp_img[:, :, 1] = NewViewDisp
    disp_img[:, :, 2] = NewViewDisp
    NewView = simple_obtainNewView(L_img, NewViewDisp, pMask).astype(np.uint8)
    return disp_img, NewView


def Disp2VS(L_img, R_img, pid, frame, fill, seman_mask=None):
    L_disp_path = 'disparity/{}_l.png'.format(pid)
    R_disp_path = 'disparity/{}_r.png'.format(pid)
    L_disp = cv2.imread(L_disp_path)
    R_disp = cv2.imread(R_disp_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if fill is True:
        path_name = 'fill'
    else:
        path_name = 'nofill'

    ViewWriter = cv2.VideoWriter(
        'Disp_View/{}_{}.avi'.format(pid, path_name), fourcc, 25, (L_disp.shape[1], L_disp.shape[0]))
    path = osp.join("New_disparity", path_name, str(pid))
    if not osp.exists(path):
        os.makedirs(path)
    DispWriter = cv2.VideoWriter(
        '{}/{}.avi'.format(path, pid), fourcc, 25, (L_disp.shape[1], L_disp.shape[0]))

    # L_disp only
    for i in range(frame):
        interp = float(i / frame)
        disp_img, NewView = Disp2SingleView(L_disp, interp, fill, seman_mask)

        cv2.imwrite('{}/{}_{}.jpg'.format(path, pid, i), disp_img)
        ViewWriter.write(NewView)
        DispWriter.write(disp_img)
        print("\r {}/{} has finished...".format(i + 1, frame), end="")
    ViewWriter.release()
    DispWriter.release()

    """
    disp_img, NewView = Disp2SingleView(L_disp, 1, fill)
    cv2.imwrite('{}_test_disp.jpg'.format(pid), disp_img)
    cv2.imwrite('{}_test_NewView.jpg'.format(pid), NewView)
    """

    print("View Synthesis finished")