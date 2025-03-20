import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import pdb
# 加载 .npy 文件
patch_data = np.load('/ailab/user/wuguocheng/Astro_SR/dataset/eval_patches/hst_9984_im_acs_wfc_total_j8mbim_drc/hst_9984_im_acs_wfc_total_j8mbim_drc_patch_0.npy', allow_pickle=True).item()
# 提取图像和掩码
image = patch_data['image']
mask = patch_data['mask']
norm = ImageNormalize(image, interval=ZScaleInterval())
# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 显示图像
axes[0].imshow(image, cmap='gray',origin='lower', norm=norm)
axes[0].set_title('Image Patch')

# 显示掩码
axes[1].imshow(mask, cmap='gray', origin='lower')
axes[1].set_title('Mask Patch')

# 显示图像
plt.savefig("/ailab/user/wuguocheng/Astro_SR/vis/vis_npy.png")