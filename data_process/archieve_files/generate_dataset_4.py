import os
import numpy as np
from tqdm import tqdm
from Astro_SR.data_process.archieve_files.psf_downsampling_3 import degrade_patch 
train_patches_dir = '/ailab/user/wuguocheng/Astro_SR/dataset/train_patches/hst_9444_01_acs_wfc_f814w_j6lk01_drc'
train_lr_dir = '/ailab/user/wuguocheng/Astro_SR/dataset/train_lr_patch'
file_txt_path = '/ailab/user/wuguocheng/Astro_SR/data_process/trainfile.txt'

os.makedirs(train_lr_dir, exist_ok=True)
scale_factor = 2
noise_level_range = [0, 5] 
file_paths = []
for root, _, files in os.walk(train_patches_dir):
    for file in tqdm(files, desc="Processing train patches"):
        if file.endswith('.npy'):
            hr_file_path = os.path.join(root, file)
            patch_data = np.load(hr_file_path, allow_pickle=True).item()
            hr_image = patch_data['image']  
            mask = patch_data['mask']     
            lr_image = degrade_patch(hr_image, mask, scale_factor, noise_level_range)
            lr_file_name = file.replace('.npy', '_lr.npy')
            lr_file_path = os.path.join(train_lr_dir, os.path.relpath(root, train_patches_dir), lr_file_name)
            os.makedirs(os.path.dirname(lr_file_path), exist_ok=True)
            np.save(lr_file_path, lr_image)
            file_paths.append(f"train,{hr_file_path},{lr_file_path}")
with open(file_txt_path, 'a') as f:
    for path in file_paths:
        f.write(path + '\n')

print("Low resolution image generation and file.txt update completed")