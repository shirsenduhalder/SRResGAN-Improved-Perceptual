import os
import random
import numpy as np
import skimage.io as sio
from tqdm import tqdm
import torch
import glob2
from natsort import natsorted

IMG_EXTENSIONS = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']
BINARY_EXTENSIONS = ['.npy']
DATASETS_BENCHMARK = ['DIV2K', 'Flickr2K', 'Set5', 'Set14', 'BSD100', 'Sun-Hays80', 'Urban100']

def is_image(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(dataroot):
    assert os.path.isdir(dataroot), '[ERROR][{}] is not a valid directory'.format(dataroot)

    image_paths = glob2.glob(dataroot, "/*")
    image_paths = [path for path in image_paths if is_image(path)]

    return image_paths