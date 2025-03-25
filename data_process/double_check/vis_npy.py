import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import pdb
# 加载 .npy 文件
patch_data = np.load('/ailab/user/zhuangguohang/ai4stronomy/Astro_SR/dataset/train_patches/hst_10802_1c_acs_wfc_total_j9r71c_drc/hst_10802_1c_acs_wfc_total_j9r71c_drc_patch_1.npy', allow_pickle=True).item()
# pdb.set_trace()
# non_zero_elements = patch_data[patch_data != 0]
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
plt.savefig("/ailab/user/zhuangguohang/ai4stronomy/Astro_SR/vis_npy.png")

patch_data = np.load('/ailab/user/zhuangguohang/ai4stronomy/Astro_SR/dataset/train_lr_patch/hst_10802_1c_acs_wfc_total_j9r71c_drc_patch_1_lr.npy', allow_pickle=True)
# pdb.set_trace()
# non_zero_elements = patch_data[patch_data != 0]
# 提取图像和掩码
image = patch_data


norm = ImageNormalize(image, interval=ZScaleInterval())
# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 显示图像
axes[0].imshow(image, cmap='gray',origin='lower', norm=norm)
axes[0].set_title('Image Patch')

# 显示掩码
# axes[1].imshow(mask, cmap='gray', origin='lower')
# axes[1].set_title('Mask Patch')

# 显示图像
plt.savefig("/ailab/user/zhuangguohang/ai4stronomy/Astro_SR/vis1_npy.png")
# import numpy as np
# import os
# from tqdm import tqdm

# def load_non_zero_elements(directory_path):
#     """生成器函数，用于逐个加载 .npy 文件中的非零元素"""
#     for f in os.listdir(directory_path):
#         if f.endswith('.npy'):
#             npy_file = os.path.join(directory_path, f)
#             patch_data = np.load(npy_file, allow_pickle=True)
#             non_zero_elements = patch_data[patch_data != 0]
#             yield non_zero_elements.flatten()

# # 定义包含 .npy 文件的目录路径
# directory_path = '/ailab/user/zhuangguohang/ai4stronomy/Astro_SR/dataset/train_lr_patch/'

# # 初始化变量用于累积总和和平方和
# total_sum = 0
# total_square_sum = 0
# total_count = 0

# # 获取目录中所有的 .npy 文件路径
# npy_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.npy')]

# # 使用生成器和 tqdm 显示进度条
# for non_zero_elements in tqdm(load_non_zero_elements(directory_path), total=len(npy_files), desc="Processing files"):
#     total_sum += np.sum(non_zero_elements)
#     total_square_sum += np.sum(non_zero_elements ** 2)
#     total_count += non_zero_elements.size

# # 计算均值和方差
# mean_value = total_sum / total_count
# variance_value = (total_square_sum / total_count) - (mean_value ** 2)

# print(f"所有非零元素的均值: {mean_value}")
# print(f"所有非零元素的方差: {variance_value}")