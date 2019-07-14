import numpy as np
import math, cv2
from sewar.full_ref import vifp, uqi

def rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    if in_img_type != np.uint8:
        img *= 255
    
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966])/255.0 + 160
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    
    return rlt.astype(in_img_type)

def ycbcr2rgb(img):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255
    
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071], [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    
    return rlt.astype(in_img_type)

#metrics 

def calc_metrics(img1, img2, crop_border, test_Y=True):
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:
        im1_in= rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError("Wrong image dimension: {}. Should be 2 or 3".format(im1_in.ndim))
    
    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    vif = calc_vif(cropped_im1 * 255, cropped_im2 * 255)
    uqi = calc_uqi(cropped_im1 * 255, cropped_im2 * 255)

    return psnr, ssim, vif, uqi
        
        

def calc_vif(img1, img2):
    return vifp(img1, img2)

def calc_uqi(img1, img2):
    return uqi(img1, img2)

def calc_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)

    if mse == 0:
        return float('inf')
    
    return 20 * math.log10(255.0/math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def calc_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')