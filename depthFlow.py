import os
import numpy as np
import cv2
from PIL import Image
from skimage.restoration import denoise_bilateral
from moviepy.editor import *
from inverse_warp import *
from skimage.io import imread, imsave
import torch
import torch.nn.functional as F


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return(b_0, b_1)


def depthFilter(disp):
    # Filter the depth
    noisy = disp
    result = denoise_bilateral(
        noisy, sigma_color=0.5, sigma_spatial=4, win_size=7, multichannel=False)
    b = np.percentile(noisy, list(range(100)))
    a = np.percentile(result, list(range(100)))
    x = estimate_coef(a, b)
    result = (result * x[1] + x[0])
    return result


def warp(rgb, depth, intrinsic, pose1, pose2):

    intrinsic = torch.squeeze(intrinsic, 1).type(torch.cuda.DoubleTensor)
    pose1 = torch.squeeze(pose1, 1).type(torch.cuda.DoubleTensor)
    pose2 = torch.squeeze(pose2, 1).type(torch.cuda.DoubleTensor)
    h, w = list(depth.size())[1:3]
    depth = depth.type(torch.cuda.DoubleTensor)
    rgb = rgb.type(torch.cuda.FloatTensor)
    # cur_depth = (10000 * torch.ones((1, h, w))).type(torch.cuda.DoubleTensor)
    warp = inverse_warp(rgb[:, :, :, :].permute(
        0, 3, 1, 2), depth, pose2, pose1, intrinsic)
    return warp


def warp_con(rgb_tensor, depth_tensor, intrinsic, pose1, pose2):

    depth_tensor = torch.from_numpy(depth_tensor)
    rgb_tensor = torch.from_numpy(rgb_tensor)
    depth_tensor = torch.unsqueeze(
        depth_tensor, 0).type(torch.cuda.DoubleTensor)
    rgb_tensor = torch.unsqueeze(rgb_tensor, 0).type(torch.cuda.DoubleTensor)

    out = warp(rgb_tensor, depth_tensor, intrinsic, pose1, pose2)
    out = (np.transpose(np.squeeze(out.cpu().numpy()),
                        (1, 2, 0))*255.0).astype(np.uint8)
    return out


def Disp2Depth(disp, fx=None, baseline=None, filter=True):
    '''
    depth = ( fx * baseline ) / disp
    ---------------------------
    Attention: fx and baseline is important for disparity transformation. And since PIL Image doesn't have
    mode for int16( 655536, which can represent 65 meter depth if we use 'millimeter' as the unit of baseline), therefore
    we use np.uint32, the mode 'I' for PIL Image.
    '''
    disp = np.array(disp)
    if filter == True:
        disp = depthFilter(disp)
    depth = fx * baseline / (disp + 0.000001)
    # depth = depth.astype(np.uint8)
    # Image.fromarray(depth).save('test_12.png')
    return depth

def getPose(dist=1, num=30):
    """
    intrinsic: [B, 3, 3]
    pose: [B, 3, 4]
    pose2 is used.
    """
    x = np.sin(np.linspace(0, 2*np.pi * (num-2) / (num-1), num)) * dist
    y = np.cos(np.linspace(0, 2*np.pi * (num-2) / (num-1), num)) * dist
    z = np.linspace(0, -dist, num)
    xv = list(x) + list(x)
    yv = list(y) + list(y)
    zv = list(z) + list(z)[::-1]

    shift = 15
    intrinsic = np.reshape(np.array([192, 0, 192, 0, 192, 192, 0, 0, 1]), [
        1, 3, 3]).astype(np.float32)
    ref_pose = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    tgt_pose = ref_pose.copy()
    tgt_pose[3] += xv[shift]
    tgt_pose[7] += yv[shift]
    tgt_pose[11] += zv[shift]
    intrinsic = np.reshape(intrinsic, [1, 3, 3])
    ref_pose = np.reshape(ref_pose, [4, 4])
    tgt_pose = np.reshape(tgt_pose, [4, 4])
    pose1 = np.reshape(np.matmul(tgt_pose, np.linalg.inv(ref_pose)), [
        1, 4, 4]).astype(np.float32)
    pose2 = np.reshape(np.matmul(ref_pose, np.linalg.inv(tgt_pose)), [
        1, 4, 4]).astype(np.float32)
    pose1 = pose1[:, :3, :]
    pose2 = pose2[:, :3, :]

    intrinsic = torch.from_numpy(intrinsic)
    pose1 = torch.from_numpy(pose1)
    pose2 = torch.from_numpy(pose2)

    return intrinsic, pose1, pose2


def getdepthFlow():
    feature_tensor = torch.from_numpy(
        np.zeros([1, depth_tensor.shape[1], depth_tensor.shape[2], 10]))
    feature = feature_tensor.type(torch.cuda.FloatTensor)
    intrinsic, pose1, pose2 = getPose()
    pid = 15
    fx = 2000
    baseline = 1000
    disp = imread("disparity/{}_l.png".format(pid))
    if len(disp.shape) != 2:
        disp = disp[:, :, 0]
    depth = Disp2Depth(disp, fx=fx, baseline=baseline)
    depth = depth.astype(np.uint32).astype(np.float32)
    depth = torch.from_numpy(depth)
    depth_tensor = torch.unsqueeze(depth, 0).type(torch.cuda.DoubleTensor)

    intrinsic = torch.squeeze(intrinsic, 1).type(torch.cuda.DoubleTensor)
    pose2 = torch.squeeze(pose2, 1).type(torch.cuda.DoubleTensor)
    depth = depth_tensor.type(torch.cuda.DoubleTensor)

    new_feature = inverse_warp(feature.permute(
        0, 3, 1, 2), depth, pose2, pose1, intrinsic)
    print(new_feature.shape)
    return new_feature

def getPose_2(id_1=57, id_2=61, id_tgt=59):
    K = np.load('Test/Ks.npy')
    R = np.load('Test/Rs.npy')
    T = np.load('Test/ts.npy')
    intrinsic = K[0, :, :]

    R_1 = R[id_1, :, :]
    T_1 = T[id_1, :]
    R_2 = R[id_2, :, :]
    T_2 = T[id_2, :]
    R_tgt = R[id_tgt, :, :]
    T_tgt = T[id_tgt, :]

    padding = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)
    extrinsix_1 = np.concatenate((R_1, np.transpose(T_1).reshape(3, 1)), axis=1)
    extrinsix_1 = np.concatenate((extrinsix_1, padding), axis=0).astype(np.float32)
    
    extrinsix_2 = np.concatenate((R_2, np.transpose(T_2).reshape(3, 1)), axis=1)
    extrinsix_2 = np.concatenate((extrinsix_2, padding), axis=0).astype(np.float32)

    extrinsix_tgt = np.concatenate((R_tgt, np.transpose(T_tgt).reshape(3, 1)), axis=1)
    extrinsix_tgt = np.concatenate((extrinsix_tgt, padding), axis=0).astype(np.float32)

    # pose1: img_1 to tgt
    pose1 = np.reshape(np.matmul(extrinsix_1, np.linalg.inv(extrinsix_tgt)), [1, 4, 4]).astype(np.float32)
    pose2 = np.reshape(np.matmul(extrinsix_2, np.linalg.inv(extrinsix_tgt)), [1, 4, 4]).astype(np.float32)
    pose1 = pose1[:, :3, :]
    pose2 = pose2[:, :3, :]

    intrinsic = torch.from_numpy(intrinsic.reshape(1, 3, 3))
    pose1 = torch.from_numpy(pose1)
    pose2 = torch.from_numpy(pose2)

    return intrinsic, pose1, pose2
    
def getdepthFlow_2(feature, depth, intrinsic, pose):
    depth = torch.from_numpy(depth)
    depth_tensor = torch.unsqueeze(depth, 0).type(torch.cuda.DoubleTensor)

    intrinsic = torch.squeeze(intrinsic, 1).type(torch.cuda.DoubleTensor)
    pose = torch.squeeze(pose, 1).type(torch.cuda.DoubleTensor)

    new_feature = inverse_warp(feature.permute(0, 3, 1, 2), depth_tensor, pose, intrinsic)
    
    return new_feature

def getDepth(id_1=57, id_2=61, id_tgt=59):
    depth_1 = np.load('Test/dm_00000057.npy')
    depth_2 = np.load('Test/dm_00000061.npy')
    depth_tgt = np.load('Test/dm_00000059.npy')
    return depth_1, depth_2, depth_tgt

def getImage(id_1=57, id_2=61, id_tgt=59):
    img_1 = cv2.imread('Test/im_00000057.jpg')
    img_2 = cv2.imread('Test/im_00000061.jpg')
    img_tgt = cv2.imread('Test/im_00000059.jpg')
    return img_1, img_2, img_tgt

if __name__ == "__main__":
    # Flow = getdepthFlow()
    img_1, img_2, img_tgt = getImage()
    depth_1, depth_2, depth_tgt = getDepth()
    # cv2.imwrite('depth.jpg', depth_tgt)
    # pose1: img_1 to img_tgt
    # pose2: img_2 to img_tgt
    intrinsic, pose1, pose2 = getPose_2()
    img_1 = torch.from_numpy(img_1.reshape(1, img_1.shape[0], img_2.shape[1], 3)).type(torch.cuda.FloatTensor)
    new_feature = getdepthFlow_2(img_1, depth_tgt, intrinsic, pose1)
    print(new_feature.shape)


