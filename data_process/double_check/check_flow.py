from astropy.io import fits
import numpy as np

# 加载原始图像
with fits.open('/ailab/group/pjlab-ai4s/ai4astro/Deep_space_explore/hst_data/jb5d08010/jb5d08010_drc.fits.gz') as hdul:
    data_original = hdul[1].data

# 加载下采样后的图像
with fits.open('/ailab/user/wuguocheng/Astro_SR/downsampled_image.fits') as hdul:
    data_reprojected = hdul[0].data

# 计算总通量（忽略 NaN 值）
S_original = np.nansum(data_original)
S_reprojected = np.nansum(data_reprojected)

# 输出结果
print(f"原始图像总通量: {S_original}")
print(f"下采样后图像总通量: {S_reprojected}")
print(f"通量差异: {S_original - S_reprojected}")