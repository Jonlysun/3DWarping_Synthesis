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


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def gauss(s):
    sigma = 10
    n = 31
    r = range(-int(n/2), int(n/2)+1)
    layer = [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2))
             for x in r]
    layer = [l/layer[15] for l in layer]
    layer[16:] = [0] * 15
    return layer


def warp(rgb, depth, intrinsic, pose1, pose2):

    intrinsic = torch.squeeze(intrinsic, 1).type(torch.cuda.DoubleTensor)
    pose1 = torch.squeeze(pose1, 1).type(torch.cuda.DoubleTensor)
    pose2 = torch.squeeze(pose2, 1).type(torch.cuda.DoubleTensor)
    h, w = list(depth.size())[1:3]
    depth = depth.type(torch.cuda.DoubleTensor)
    rgb = rgb.type(torch.cuda.FloatTensor)
    # cur_depth = (10000 * torch.ones((1, h, w))).type(torch.cuda.DoubleTensor)
    warp = inverse_warp(rgb[:, :, :, :].permute(
        0, 3, 1, 2), depth, pose2, intrinsic)
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


def generateKenVideo(img, depth, pid, dist=1, num=30):
    img = np.array(img / 255.0).astype(np.float32)
    depth = np.array(depth).astype(np.float32)
    x = list(np.linspace(-dist, 0, num)) + list(np.linspace(0, dist, num))
    y = [0] * 60
    z = list(np.linspace(-dist, 0, num)) + list(np.linspace(0, -dist, num))
    xv = list(x) + list(x)[::-1]
    yv = list(y) + list(y)[::-1]
    zv = list(z) + list(z)[::-1]

    images = []
    for shift in range(len(xv)):
        intrinsic = np.reshape(np.array([128, 0, 128, 0, 128, 128, 0, 0, 1]), [
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
        out = warp_con(img, depth, intrinsic, pose1, pose2)
        images.append(out)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ViewWriter = cv2.VideoWriter(
        'Depth_View/{}_ken_{}.avi'.format(pid, dist), fourcc, 25, (img.shape[1], img.shape[0]))
    for i, img in enumerate(images):
        imsave('Test/{}.jpg'.format(i), img)
        r,g,b = cv2.split(img)
        img = cv2.merge([b, g, r])
        ViewWriter.write(img)
    ViewWriter.release()

def generateCircleVideo(img, depth, pid, dist=1, num=30):
    img = np.array(img / 255.0).astype(np.float32)
    depth = np.array(depth).astype(np.float32)
    x = np.sin(np.linspace(0, 2*np.pi * (num-2) / (num-1), num)) * dist
    y = np.cos(np.linspace(0, 2*np.pi * (num-2) / (num-1), num)) * dist
    z = np.linspace(0, -dist, num)
    xv = list(x) + list(x)
    yv = list(y) + list(y)
    zv = list(z) + list(z)[::-1]

    images = []
    for shift in range(len(xv)):
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
        out = warp_con(img, depth, intrinsic, pose1, pose2)
        images.append(out)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ViewWriter = cv2.VideoWriter(
        'Depth_View/{}_cir_{}.avi'.format(pid, dist), fourcc, 25, (img.shape[1], img.shape[0]))
    for i, img in enumerate(images):
        imsave('Test/{}.jpg'.format(i), img)
        r,g,b = cv2.split(img)
        img = cv2.merge([b, g, r])
        ViewWriter.write(img)
    ViewWriter.release()


def Depth2VS(L_img, pid, fx, baseline, fill=True):
    '''
    the accuracy of depth depend on fx and baseline
    '''
    disp = imread("disparity/{}_l.png".format(pid))
    if len(disp.shape) != 2:
        disp = disp[:, :, 0]
    depth = Disp2Depth(disp, fx=fx, baseline=baseline)
    depth = depth.astype(np.uint32)
    # Image.fromarray(depth.astype(np.uint8)).save('depth/{}_l.png'.format(pid))
    imsave('depth/{}_l.png'.format(pid), depth.astype(np.uint8))
    if depth.max() < 3000:
        imsave('depth/{}_l.png'.format(pid), depth.astype(np.uint8))
    else:
        imsave('depth/{}_l.png'.format(pid), depth.astype(np.uint16))
    generateKenVideo(L_img, depth, pid=pid, dist=10.0)
    generateCircleVideo(L_img, depth, pid=pid, dist=10.0)

if __name__ == '__main__':
    # fx=4000, baselien=300 for pid 11
    # fx=200, baselien=10000 for pid 12
    pid = 15
    L_img = imread('image/{}_l.jpg'.format(pid))
    Depth2VS(L_img, pid=pid, fx=200, baseline=10000)
