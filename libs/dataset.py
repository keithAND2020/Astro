import os
import pdb
import numpy as np
import torch.utils.data as data
import torch
from torch.nn.functional import avg_pool2d, max_pool2d
import torchvision.transforms as transforms
from skimage.util.shape import view_as_windows
from einops import rearrange
import glob
import json
from astropy.io import fits
from reproject import reproject_exact
import cv2
from astropy.wcs import WCS
import warnings
from scipy.ndimage import zoom
from scipy.signal import convolve2d
from scipy.special import j1

warnings.filterwarnings('ignore', category=UserWarning)

class RFI_MultiInput(data.Dataset):
    def __init__(self,
                 split,
                 root_dir,
                 **kargs):

        self.path = root_dir

        if split == 'train':
            self.data_list = np.load("/home/bingxing2/ailab/scxlab0063/AstroIR/dataset/train.npy")

        if split == 'eval':
            # self.data_list = glob.glob(self.path + "train/*.npy")[:]
            self.data_list = np.load("/home/bingxing2/ailab/scxlab0063/AstroIR/dataset/eval.npy")


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        target = self.data_list[index]

        meann = np.mean(target)
        stdd = np.std(target)
        target = (target - np.mean(target)) / np.std(target)

        data = self.apply_psf(target)
        # scale_factor = (128 / 256, 128 / 256)
        # data = zoom(data, scale_factor, order=1)

        data = torch.from_numpy(data).float().unsqueeze(0)
        target = torch.from_numpy(target).float().unsqueeze(0)

        return data, target, (meann, stdd)
        # return data, target

    def apply_psf(self, image):
        sigma = 1
        size = 64
        x, y = np.ogrid[-size // 2:size // 2, -size // 2:size // 2]
        r = np.hypot(x, y) + 1e-10
        airy = (2 * j1(r) / r) ** 2
        airy /= airy.sum()
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        atmospheric_turbulence = g / g.sum()
        psf = airy * atmospheric_turbulence
        psf = psf / psf.sum()
        return convolve2d(image, psf, mode='same', boundary='wrap')

