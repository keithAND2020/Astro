import json
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (ZScaleInterval, ImageNormalize)
from scipy.ndimage import binary_erosion, generate_binary_structure
import pdb
import numpy as np
# with open("/ailab/user/wuguocheng/Astro_SR/data_process/split_file/train.json", "r") as f:
#     train_files = json.load(f)

# fits_filepath = train_files[4]
def load_data(file_path):
    with fits.open(file_path) as hdul:
        img_data = hdul[1].data.astype(float)  
        zero_mask = (img_data == 0)
        structure = generate_binary_structure(2, 1)  # 3x3 结构元素
        eroded_zero_mask = binary_erosion(zero_mask, structure=structure)
        img_data[eroded_zero_mask] = np.nan

        # 步骤 4：生成掩码
        mask = ~np.isnan(img_data)

        return img_data, mask
fits_filepath = "/ailab/user/wuguocheng/Astro_SR/dataset/psf_downsampled/jd8f28020_drc_downsampled.fits"

hdu = fits.open(fits_filepath)[0]
print(fits.info(fits_filepath))
img_data = hdu.data
# 

norm = ImageNormalize(img_data, interval=ZScaleInterval())

plt.figure(figsize=(16, 16))
plt.imshow(img_data, cmap='gray', norm=norm)
plt.colorbar(label='Intensity')
plt.title(f"FITS Image: {fits_filepath}")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.savefig('/ailab/user/wuguocheng/Astro_SR/vis/result_fits_3.png')