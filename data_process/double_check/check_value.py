import numpy as np
import astropy.units as u
import pdb
from tqdm import tqdm
import os
train_patches_dir = '/ailab/user/wuguocheng/Astro_SR/dataset/train_patches/jd8f28020_drc'
zero_image_files = []
npy_files = []
for root, dirs, files in os.walk(train_patches_dir):
    for file in files:
        if file.endswith('.npy'):
            npy_files.append(os.path.join(root, file))
for file_path in tqdm(npy_files, desc="Checking .npy files"):
    try:
        patch_data = np.load(file_path, allow_pickle=True).item()
        image = patch_data['image']
        if np.all(image == 0):
            zero_image_files.append(file_path)
    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")
if zero_image_files:
    print("以下文件中的 image 全为 0:")
    for file in zero_image_files:
        print(file)
else:
    print("没有文件中的 image 全为 0。")