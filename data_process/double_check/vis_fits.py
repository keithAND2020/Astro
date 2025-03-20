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
    # 步骤 1：加载图像数据
    with fits.open(file_path) as hdul:
        img_data = hdul[1].data.astype(float)  # 确保数据类型为浮点数，支持 NaN

        # 步骤 2：检测连续全 0 区域
        zero_mask = (img_data == 0)

        # 使用形态学腐蚀操作，识别连续全 0 区域
        structure = generate_binary_structure(2, 1)  # 3x3 结构元素
        eroded_zero_mask = binary_erosion(zero_mask, structure=structure)

        # 步骤 3：将连续全 0 区域转换为 NaN
        # eroded_zero_mask 为 True 的地方表示该像素及其 3x3 邻域内全为 0
        img_data[eroded_zero_mask] = np.nan

        # 步骤 4：生成掩码
        mask = ~np.isnan(img_data)

        return img_data, mask
fits_filepath = "/ailab/group/pjlab-ai4s/ai4astro/Deep_space_explore/hst_data/hst_12925_09_acs_wfc_total_jbyq09/hst_12925_09_acs_wfc_total_jbyq09_drc.fits.gz"
hdu = fits.open(fits_filepath)[1]

print(fits.info(fits_filepath))
img_data = hdu.data
# 

norm = ImageNormalize(img_data, interval=ZScaleInterval())

plt.figure(figsize=(16, 16))
plt.imshow(img_data, cmap='gray', origin='lower', norm=norm)
plt.colorbar(label='Intensity')
plt.title(f"FITS Image: {fits_filepath}")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.savefig('/ailab/user/wuguocheng/Astro_SR/vis/result_fits2.png')