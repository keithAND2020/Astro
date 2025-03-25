import os
import json
import numpy as np
from shapely.wkt import loads
from astropy.io import fits
from tqdm import tqdm
from scipy.ndimage import binary_erosion, generate_binary_structure
import pdb
with open("/ailab/user/zhuangguohang/ai4stronomy/Astro_SR/data_process/split_file/datasetlist.txt", "r") as f:
    lines = f.readlines()
filtered_data = []
total_before = len(lines)

for line in tqdm(lines):
    try:
        fits_filepath, wkt_str = line.strip().split(":")
        with fits.open(fits_filepath) as hdul:
            header = hdul[1].header
            ncombine = header.get('NCOMBINE', 0)
            if ncombine == 4:
                filtered_data.append((fits_filepath, wkt_str))
    except Exception as e:
        print(f"Processing {fits_filepath} fail: {e}")

total_after = len(filtered_data)
print(f"{total_before}")
print(f"{total_after}")
train_files = []
eval_files = []

for fits_filepath, wkt_str in filtered_data:
    try:
        polygon = loads(wkt_str)
        ra_values = [point[0] for point in polygon.exterior.coords]
        min_ra = min(ra_values)
        max_ra = max(ra_values)
        if max_ra < 250:
            train_files.append(fits_filepath)
        elif min_ra > 255:
            eval_files.append(fits_filepath)
    except Exception as e:
        print(f"processing {fits_filepath} failed: {e}")
def patchify(image, mask, patch_size, stride, useful_region_th=0.8):
    patches = []
    h, w = image.shape
    
    for x_idx in range(0, h - patch_size + 1, stride):
        for y_idx in range(0, w - patch_size + 1, stride):
            image_patch = image[x_idx:x_idx + patch_size, y_idx:y_idx + patch_size]
            mask_patch = mask[x_idx:x_idx + patch_size, y_idx:y_idx + patch_size]

            if (mask_patch.mean() > useful_region_th and 
                image_patch.shape[0] == patch_size and 
                image_patch.shape[1] == patch_size):
                coordinate = [x_idx, y_idx]
                patches.append((image_patch, mask_patch, coordinate))
    return patches
def load_data(file_path):
    with fits.open(file_path) as hdul:
        img_data = hdul[1].data.astype(float)  
        zero_mask = (img_data == 0)
        structure = generate_binary_structure(2, 1)  
        eroded_zero_mask = binary_erosion(zero_mask, structure=structure)

        img_data[eroded_zero_mask] = np.nan
        mask = ~np.isnan(img_data)
        return img_data, mask

def process_and_save_patches(fits_filepath, output_dir, patch_size, stride, useful_region_th=0.8):
    try:
        image, mask = load_data(fits_filepath)
        shape = list(image.shape)
        patches = patchify(image, mask, patch_size, stride, useful_region_th)
        basename = os.path.basename(fits_filepath).replace(".fits", "").replace(".gz", "")
        patch_dir = os.path.join(output_dir, basename)
        os.makedirs(patch_dir, exist_ok=True)
        meta_json = {
            "shape": shape,
            "original_path": fits_filepath
        }
        meta_json_filepath = os.path.join(patch_dir, "meta.json")
        with open(meta_json_filepath, "w") as json_file:
            json.dump(meta_json, json_file, indent=4)
        
        zero_image_files = []
        
        for idx, (image_patch, mask_patch, coordinate) in tqdm(enumerate(patches), total=len(patches), desc=f"Saving patches for {basename}"):
            if np.all(image_patch == 0):
                zero_image_files.append(f"{basename}_patch_{idx}.npy")
            
            patch_data = {
                "image": image_patch,
                "mask": mask_patch,
                "coord": coordinate
            }
            patch_filename = f"{basename}_patch_{idx}.npy"
            patch_filepath = os.path.join(patch_dir, patch_filename)
            np.save(patch_filepath, patch_data)
        
        print(f"Processed {len(patches)} patches to {patch_dir}")
        if zero_image_files:
            print("以下 patch 文件的 image 全为 0:")
            for file in zero_image_files:
                print(file)
        else:
            print("没有 patch 的 image 全为 0。")
    
    except Exception as e:
        print(f"Processing {fits_filepath} failed: {e}")

patch_size = 256
stride = patch_size // 2
train_patch_dir = "/ailab/user/zhuangguohang/ai4stronomy/Astro_SR/dataset/train_patches"
eval_patch_dir = "/ailab/user/zhuangguohang/ai4stronomy/Astro_SR/dataset/eval_patches"
os.makedirs(train_patch_dir, exist_ok=True)
os.makedirs(eval_patch_dir, exist_ok=True)



for fits_filepath in tqdm(train_files, desc="Processing train files"):
    process_and_save_patches(fits_filepath, train_patch_dir, patch_size=patch_size, stride=stride)

for fits_filepath in tqdm(eval_files, desc="Processing eval files"):
    process_and_save_patches(fits_filepath, eval_patch_dir, patch_size=patch_size, stride=stride)

print("All done")