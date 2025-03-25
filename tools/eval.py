from model import SwinIR
import torch

import numpy as np
from imageio import imread
from astropy.stats import sigma_clipped_stats
from photutils.aperture import aperture_photometry, CircularAperture
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.psf import PSFPhotometry
from photutils.psf import CircularGaussianPRF
from astropy.io import fits
from astropy.table import QTable, Table
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import pandas as pd
import pdb
from photutils.psf import SourceGrouper
from matplotlib.patches import Circle
from astropy.visualization import (ZScaleInterval, ImageNormalize)
from scipy.ndimage import zoom
from matplotlib import pyplot as plt
import astropy.io.fits as pyfits
from astropy.visualization import ZScaleInterval
from matplotlib.colors import Normalize
from PIL import Image
from scipy.signal import convolve2d
from scipy.special import j1


def vis(array, norm):
    fig, ax = plt.subplots()
    cax = ax.imshow(array,
                    aspect='auto',
                    rasterized=True,
                    interpolation='nearest',
                    cmap='gray',
                    norm=norm
                    )
    fig.colorbar(cax, ax=ax)
    plt.show()
    plt.close()
def fits_vis(ori_array):
    z = ZScaleInterval(n_samples=1000, contrast=0.25)
    z1, z2 = z.get_limits(ori_array)  # 19个一起统计中位数 、 方差
    norm = Normalize(vmin=z1, vmax=z2)
    normalized_array = norm(ori_array)
    cmap = plt.get_cmap('gray')
    wave_array = cmap(normalized_array)
    wave_array = (wave_array[..., 0] * 255).astype(np.uint8)
    return wave_array.astype(np.uint8)
def apply_psf(image):
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

# filename = "C:/Users/11068/Desktop/上海AI Lab/AILab/hst_skycell-p2433x07y01_acs_wfc_f606w_all_drc.fits"
# hdul = fits.open(filename)
#
# data = hdul[1].data
# header = hdul[1].header
# data = np.nan_to_num(data)
#
# rows, cols = np.nonzero(data)
# min_row, min_col = min(rows), min(cols)
# max_row, max_col = max(rows), max(cols)
# data = data[min_row:max_row + 200, min_col:max_col+1]
# norm = ImageNormalize(data, interval=ZScaleInterval())  # plt.imshow(data, origin='lower', cmap='gray', norm=norm)
# scale_factor = (256 / data.shape[0], 256 / data.shape[1])
# data = zoom(data, scale_factor, order=1)

import glob
data_list = glob.glob("C:/Users/11068/Desktop/上海AI Lab/AILab/eval_data/*.npy")

data = np.load(data_list[0])
norm = ImageNormalize(data, interval=ZScaleInterval())
ori = data.copy()

box_size = 16  # 视情况可调
bkg_estimator = MedianBackground()
bkg1 = Background2D(data, box_size, filter_size=(3,3), bkg_estimator=bkg_estimator)
data_sub = data - bkg1.background
GT = data_sub.copy()
mean_val, median_val, std_val = sigma_clipped_stats(data_sub, sigma=3.0)
threshold = 11.0 * std_val  # 可根据需求调节，例如3~8倍std
fwhm_guess = 1.0           # 对HST/ACS来说一般几像素左右，具体需调参
daofind = DAOStarFinder(threshold=threshold, fwhm=fwhm_guess)
sources_tbl = daofind(data_sub)
sources = sources_tbl.to_pandas()
positions = np.transpose((sources['xcentroid'].values, sources['ycentroid'].values))
aperture_radius = 10.0
apertures = CircularAperture(positions, r=aperture_radius)
phot_table1 = aperture_photometry(data_sub, apertures)
result1 = sum(np.abs(phot_table1["aperture_sum"]))


model = SwinIR(img_size=256, patch_size=1, in_chans=1, out_chans=1,
             embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
             window_size=8, mlp_ratio=2., upscale=1, img_range=1.,
             upsampler='pixelshuffledirect', resi_connection='1conv', scale=1,)

model.load_state_dict(torch.load("best.pth", map_location=torch.device('cpu')))
meann = np.mean(ori)
stdn = np.std(ori)
ori = (ori - meann) / stdn
pred = model(torch.tensor(apply_psf(ori)).float().unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
pred = np.array(pred.detach().numpy())
print(pred.shape)

pred = pred * stdn + meann

box_size = 16  # 视情况可调
bkg_estimator = MedianBackground()
bkg = Background2D(pred, box_size, filter_size=(3,3), bkg_estimator=bkg_estimator)
data_sub = pred - bkg.background
Pred = data_sub.copy()

mean_val, median_val, std_val = sigma_clipped_stats(data_sub, sigma=3.0)
threshold = 11.0 * std_val  # 可根据需求调节，例如3~8倍std
fwhm_guess = 1.0           # 对HST/ACS来说一般几像素左右，具体需调参
daofind = DAOStarFinder(threshold=threshold, fwhm=fwhm_guess)
sources_tbl = daofind(data_sub)
sources = sources_tbl.to_pandas()
positions = np.transpose((sources['xcentroid'].values, sources['ycentroid'].values))
apertures = CircularAperture(positions, r=aperture_radius)
phot_table2 = aperture_photometry(data_sub, apertures)
result2 = sum(np.abs(phot_table2["aperture_sum"]))

print(f"GT: {result1:.4f}")
print(f"Pred: {result2:.4f}")
import pandas as pd
series1 = pd.Series(phot_table1["aperture_sum"])
series2 = pd.Series(phot_table2["aperture_sum"])
aligned_series1, aligned_series2 = series1.align(series2, join='inner')
tfe = np.sum((np.abs(aligned_series1.values) - np.abs(aligned_series2.values))) / np.sum(np.abs(aligned_series1.values))
print(f"TFE: {tfe:.4f}")
