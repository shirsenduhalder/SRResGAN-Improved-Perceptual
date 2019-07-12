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
'''
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
'''

def PSNR(pred, gt):
    return compare_psnr(pred, gt)

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
avg_psnr_bicubic = 0.0
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

    #im_h_matlab = matlab.double((im_h / 255.).tolist())
    
    
    # im_h_matlab = double(im_h / 255.)
    
    
    # print(im_h_matlab)
    # im_h  = convert(im_h_matlab)
    # print(im_h)
    #im_h_ycbcr = eng.rgb2ycbcr(im_h_matlab)
    
    
    # im_h_ycbcr = rgb2ycbcr(im_h_matlab, only_y=False)*255
    
    
    # im_h_ycbcr = np.array(im_h_ycbcr).reshape(im_h_ycbcr.size, order='F').astype(np.float32) * 255.
    # print(im_h_ycbcr.shape)
    
    im_b = cv2.resize(im_l, (im_h.shape[1], im_h.shape[0]), cv2.INTER_CUBIC).astype(np.uint8)
    # im_b_y = cv2.cvtColor(im_b, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    # im_gt_y = cv2.cvtColor(im_gt, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    psnr_bicubic = PSNR(im_b, im_gt)
    avg_psnr_bicubic += psnr_bicubic
    psnr_predicted = PSNR(im_h, im_gt)
    avg_psnr_predicted += psnr_predicted

    img_save = np.vstack([im_h, im_gt])
    save_folder = os.path.join(image_folder.replace('test', 'test_samples'), opt.model.split('/')[-2])

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    sio.imsave(os.path.join(save_folder, img_hr.split('/')[-1].replace('_HR', '')), img_save)


print("Scale=", opt.scale_factor)
print("Dataset=", opt.test_folder.split('/')[-1])
print("PSNR_predicted=", avg_psnr_predicted/len(image_list_hr))
print("PSNR_bicubic=", avg_psnr_bicubic/len(image_list_hr))
print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list_hr)))