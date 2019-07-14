import os, glob2
from natsort import natsorted
import numpy as np
import skimage.io as sio
from utils import *
import argparse

COMPARISON_ROOT = "comparison_methods"

methods = natsorted(os.listdir(COMPARISON_ROOT))
methods = [method for method in methods if os.path.isdir(os.path.join(COMPARISON_ROOT, method)) and method != "HR"]

methods_root = natsorted([os.path.join(COMPARISON_ROOT, folder) for folder in methods])

gt_root = os.path.join(COMPARISON_ROOT, 'HR')

parser = argparse.ArgumentParser()
parser.add_argument("--set", help="Enter the set of test images", default="Set14")
parser.add_argument("--scale", help="Enter scale of SR Default: 4", default=4, type=int)

opt = parser.parse_args()

gt_folder = os.path.join(gt_root, [s for s in os.listdir(gt_root) if opt.set.lower() in s.lower()][0])

gt_images = natsorted(glob2.glob(gt_folder + "/*.*"))

len_images = len(gt_images)

for method, method_root in zip(methods, methods_root):

    testset_folder = os.path.join(method_root, [s for s in os.listdir(method_root) if opt.set.lower() in s.lower()][0])

    method_images = natsorted(glob2.glob(testset_folder + "/*.*"))
    
    assert len(method_images) == len_images, "Number of HR and SR images to compare do not match"

    avg_psnr, avg_ssim, avg_vif, avg_uqi = 0, 0, 0, 0

    for gt, pred in zip(gt_images, method_images):
        img_gt = sio.imread(gt)
        if img_gt.ndim == 2:
            img_gt = np.tile(np.expand_dims(img_gt, axis=-1), (1, 1, 3))
        img_pred = sio.imread(pred)

        psnr, ssim, vif, uqi = calc_metrics(img_gt, img_pred, opt.scale)

        avg_psnr += psnr
        avg_ssim += ssim
        avg_vif += vif
        avg_uqi += uqi
    
    avg_psnr, avg_ssim, avg_vif, avg_uqi = avg_psnr/len_images, avg_ssim/len_images, avg_vif/len_images, avg_uqi/len_images
    print("METHOD: {}".format(method))
    print("PSNR: {}, SSIM: {}, VIF: {}, UQI:{})".format(avg_psnr, avg_ssim, avg_vif, avg_uqi))