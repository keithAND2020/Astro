import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_erosion, generate_binary_structure
from tqdm import tqdm
import pdb
def load_data(file_path, hr=True):
    """加载 FITS 文件，返回图像数据和掩码"""
    with fits.open(file_path) as hdul:
        if hr:
            img_data = hdul[1].data.astype(float)
        else:
            img_data = hdul[0].data.astype(float)
        zero_mask = (img_data == 0)
        structure = generate_binary_structure(2, 1)
        eroded_zero_mask = binary_erosion(zero_mask, structure=structure)
        img_data[eroded_zero_mask] = np.nan
        mask = ~np.isnan(img_data)
        return img_data, mask

def patchify(image, mask, patch_size, stride, useful_region_th=0.8):
    """将图像分割为 patch，并筛选有效区域"""
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

def generate_dataloader_txt(hr_patches, lr_patches, hr_patch_dir, lr_patch_dir, dataloader_txt, identifier):
    """保存 HR 和 LR patch 并生成 dataloader.txt"""
    with open(dataloader_txt, "a") as f:
        for idx, (hr_patch, lr_patch) in enumerate(zip(hr_patches, lr_patches)):
            hr_image_patch, hr_mask_patch, hr_coord = hr_patch
            lr_image_patch, lr_mask_patch, lr_coord = lr_patch
            hr_patch_filename = f"{identifier}_hr_patch_{idx}.npy"
            hr_patch_path = os.path.join(hr_patch_dir, hr_patch_filename)
            np.save(hr_patch_path, {"image": hr_image_patch, "mask": hr_mask_patch, "coord": hr_coord})
            
            lr_patch_filename = f"{identifier}_lr_patch_{idx}.npy"
            lr_patch_path = os.path.join(lr_patch_dir, lr_patch_filename)
            np.save(lr_patch_path, {"image": lr_image_patch, "mask": lr_mask_patch, "coord": lr_coord})
            
            f.write(f"{hr_patch_path},{lr_patch_path},{hr_coord}\n")

def process_patchify(train_files_path, eval_files_path, dataset_dir, dataload_filename_dir, hr_patch_size=256, lr_patch_size=128, stride=128, useful_region_th=0.8, scale_factor=2):
    """对 HR 和 LR 图像进行 Patchify 并生成训练集和验证集的 dataloader.txt"""
    train_hr_patch_dir = os.path.join(dataset_dir, "train_hr_patch")
    train_lr_patch_dir = os.path.join(dataset_dir, "train_lr_patch")
    eval_hr_patch_dir = os.path.join(dataset_dir, "eval_hr_patch")
    eval_lr_patch_dir = os.path.join(dataset_dir, "eval_lr_patch")
    os.makedirs(train_hr_patch_dir, exist_ok=True)
    os.makedirs(train_lr_patch_dir, exist_ok=True)
    os.makedirs(eval_hr_patch_dir, exist_ok=True)
    os.makedirs(eval_lr_patch_dir, exist_ok=True)
    
    os.makedirs(dataload_filename_dir, exist_ok=True)
    train_dataloader_txt = os.path.join(dataload_filename_dir, "train_dataloader.txt")
    eval_dataloader_txt = os.path.join(dataload_filename_dir, "eval_dataloader.txt")
    if os.path.exists(train_dataloader_txt):
        os.remove(train_dataloader_txt)
    if os.path.exists(eval_dataloader_txt):
        os.remove(eval_dataloader_txt)

    with open(train_files_path, "r") as f:
        train_files = [line.strip().split(',') for line in f.readlines()]
    for hr_path, lr_path in tqdm(train_files, desc="Processing train files"):
        try:
            hr_image, hr_mask = load_data(hr_path, hr=True)
            lr_image, lr_mask = load_data(lr_path, hr=False)
            identifier = os.path.basename(hr_path).replace(".fits", "").replace(".gz", "")
            
            hr_patches = patchify(hr_image, hr_mask, hr_patch_size, stride, useful_region_th)
            lr_patches = patchify(lr_image, lr_mask, lr_patch_size, stride // scale_factor, useful_region_th)
            
            if len(hr_patches) > 0 and len(lr_patches) > 0:
                generate_dataloader_txt(hr_patches, lr_patches, train_hr_patch_dir, train_lr_patch_dir, train_dataloader_txt, identifier)
                pdb.set_trace()
            else:
                print(f"warning: {hr_path} No valid patch")
        except Exception as e:
            print(f"processing  {hr_path} fail: {e}")
    
    with open(eval_files_path, "r") as f:
        eval_files = [line.strip().split(',') for line in f.readlines()]
    for hr_path, lr_path in tqdm(eval_files, desc="Processing eval files"):
        try:
            hr_image, hr_mask = load_data(hr_path, hr=True)
            lr_image, lr_mask = load_data(lr_path, hr=False)
            identifier = os.path.basename(hr_path).replace(".fits", "").replace(".gz", "")
            
            hr_patches = patchify(hr_image, hr_mask, hr_patch_size, stride, useful_region_th)
            lr_patches = patchify(lr_image, lr_mask, lr_patch_size, stride // scale_factor, useful_region_th)
            
            if len(hr_patches) > 0 and len(lr_patches) > 0:
                generate_dataloader_txt(hr_patches, lr_patches, eval_hr_patch_dir, eval_lr_patch_dir, eval_dataloader_txt, identifier)
            else:
                print(f"warning: {hr_path} No valid patch")
        except Exception as e:
            print(f"processing  {hr_path} fail: {e}")

if __name__ == "__main__":
    train_files_path = "/ailab/user/wuguocheng/Astro_SR/data_process/split_file/train_files.txt"
    eval_files_path = "/ailab/user/wuguocheng/Astro_SR/data_process/split_file/eval_files.txt"
    dataset_dir = "../dataset"
    dataload_filename_dir = "../dataload_filename"
    process_patchify(train_files_path, eval_files_path, dataset_dir, dataload_filename_dir)