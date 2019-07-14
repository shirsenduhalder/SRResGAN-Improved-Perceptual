#import matlab.engine
import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
from numpy import *
import time, math, glob2
import skimage.io as sio
import cv2
from math import log10,floor 
from natsort import natsorted
from skimage.measure import compare_psnr
import matplotlib.pyplot as plt
from utils import *

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/model_srresnet.pth", type=str, help="model path")
parser.add_argument("--test_folder", default="data/test/Set14", help="Enter test dataset folder")
# parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--scale_factor", default=4, type=int, help="scale factor, Default: 4, Options: {2, 3, 4}")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    '''
    in_img_type = img.dtype
    img.astype(np.float64)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


opt = parser.parse_args()
cuda = opt.cuda
#eng = matlab.engine.start_matlab()

assert opt.scale_factor in [2, 3, 4], "Scale factor not supported"

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model)['model']

image_folder =  os.path.join(opt.test_folder, 'image_SRF_{}'.format(opt.scale_factor))

image_list_hr = natsorted(glob2.glob(image_folder + "/*HR*.*"))
image_list_lr = natsorted(glob2.glob(image_folder + "/*LR*.*"))

avg_psnr_predicted = 0.0
avg_ssim_predicted = 0.0
avg_vif_predicted = 0.0
avg_uqi_predicted = 0.0

avg_psnr_bicubic = 0.0
avg_ssim_bicubic = 0.0
avg_vif_bicubic = 0.0
avg_uqi_bicubic = 0.0

avg_elapsed_time = 0.0

for img_hr, img_lr in zip(image_list_hr, image_list_lr):
    print("Processing ", img_hr)

    im_gt = sio.imread(img_hr)
    im_l = sio.imread(img_lr)

    if im_gt.ndim == 2:
        im_gt = np.tile(np.expand_dims(im_gt, axis=-1), (1, 1, 3))
    
    if im_l.ndim == 2:
        im_l = np.tile(np.expand_dims(im_l, axis=-1), (1, 1, 3))

    im_input = im_l.astype(np.float32).transpose(2,0,1)
    im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
    im_input = Variable(torch.from_numpy(im_input/255.0).float())

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    HR_4x = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    HR_4x = HR_4x.cpu()

    im_h = HR_4x.data[0].numpy().astype(np.float32)
    im_h = im_h*255
    im_h = np.clip(im_h, 0, 255)  
    im_h = im_h.transpose(1,2,0).astype(np.uint8)

    
    im_b = cv2.resize(im_l, (im_h.shape[1], im_h.shape[0])).astype(np.uint8)
    
    bi_psnr, bi_ssim, bi_vif, bi_uqi = calc_metrics(im_b, im_gt, opt.scale_factor)
    pred_psnr, pred_ssim, pred_vif, pred_uqi = calc_metrics(im_h, im_gt, opt.scale_factor)

    avg_psnr_predicted += pred_psnr
    avg_ssim_predicted += pred_ssim
    avg_vif_predicted += pred_vif
    avg_uqi_predicted += pred_uqi

    avg_psnr_bicubic += bi_psnr
    avg_ssim_bicubic += bi_ssim
    avg_vif_bicubic += bi_vif
    avg_uqi_bicubic += bi_uqi

    save_folder = os.path.join(image_folder.replace('test', 'test_samples'), opt.model.split('/')[-2])

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    sio.imsave(os.path.join(save_folder, img_hr.split('/')[-1].replace('_HR', '')), im_h)


# print("Scale=", opt.scale_factor)
print("Dataset=", opt.test_folder.split('/')[-1])

print("PSNR_predicted=", avg_psnr_predicted/len(image_list_hr))
print("PSNR_bicubic=", avg_psnr_bicubic/len(image_list_hr))

print("SSIM_predicted=", avg_ssim_predicted/len(image_list_hr))
print("SSIM_bicubic=", avg_ssim_bicubic/len(image_list_hr))


print("VIF_predicted=", avg_vif_predicted/len(image_list_hr))
print("VIF_bicubic=", avg_vif_bicubic/len(image_list_hr))

print("UQI_predicted=", avg_uqi_predicted/len(image_list_hr))
print("UQI_bicubic=", avg_uqi_bicubic/len(image_list_hr))
# print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list_hr)))