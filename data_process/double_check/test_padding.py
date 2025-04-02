import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_erosion, generate_binary_structure
from reproject import reproject_exact
from scipy.ndimage import zoom

def load_fits(file_path, hr=True):
    """加载 FITS 文件，返回图像数据、掩码和 WCS 信息"""
    with fits.open(file_path) as hdul:
        if hr:
            img_data = hdul[1].data.astype(float)  # 高分辨率图像从 HDU 1 读取
        else:
            img_data = hdul[0].data.astype(float)  # 低分辨率图像从 HDU 0 读取
        wcs = WCS(hdul[0].header) if not hr else WCS(hdul[1].header)
        
        # 生成掩码
        zero_mask = (img_data == 0)  # 标记值为 0 的区域
        structure = generate_binary_structure(2, 1)  # 定义腐蚀结构
        eroded_zero_mask = binary_erosion(zero_mask, structure=structure)  # 腐蚀操作
        img_data[eroded_zero_mask] = np.nan  # 将无效区域标记为 NaN
        mask = ~np.isnan(img_data)  # 生成掩码，有效区域为 True
        return img_data, mask, wcs

def pad_to_multiple(image, wcs, multiple=256, pad_value=np.nan):
    """将图像 padding 到指定倍数，并更新 WCS"""
    h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_value)
    padded_wcs = wcs.deepcopy()
    padded_wcs.array_shape = padded_image.shape
    return padded_image, padded_wcs

def downsample_image(image, wcs, scale_factor=2):
    """对图像进行下采样并更新 WCS"""
    target_shape = (int(image.shape[0] / scale_factor), int(image.shape[1] / scale_factor))
    target_wcs = wcs.deepcopy()
    target_wcs.wcs.crpix = [crpix / scale_factor for crpix in wcs.wcs.crpix]
    if hasattr(wcs.wcs, 'cd'):
        target_wcs.wcs.cd = wcs.wcs.cd * scale_factor
    else:
        target_wcs.wcs.cdelt = [cdelt * scale_factor for cdelt in wcs.wcs.cdelt]
    downsampled_image, _ = reproject_exact((image, wcs), target_wcs, shape_out=target_shape)
    return downsampled_image, target_wcs

def patchify_hr(image, mask, patch_size=256, stride=128, useful_region_th=0.8):
    """对 HR 图像进行 Patchify，返回满足阈值的 patch"""
    patches = []
    h, w = image.shape
    for x in range(0, h - patch_size + 1, stride):
        for y in range(0, w - patch_size + 1, stride):
            mask_patch = mask[x:x + patch_size, y:y + patch_size]
            if mask_patch.mean() > useful_region_th:
                image_patch = image[x:x + patch_size, y:y + patch_size]
                patches.append((image_patch, mask_patch, (x, y)))
    return patches

def get_lr_patch(lr_image, lr_mask, hr_coord, scale_factor=2, lr_patch_size=128, useful_region_th=0.8):
    """根据 HR patch 坐标从 LR 图像中提取并筛选 LR patch"""
    x_hr, y_hr = hr_coord
    x_lr = x_hr // scale_factor
    y_lr = y_hr // scale_factor
    lr_mask_patch = lr_mask[x_lr:x_lr + lr_patch_size, y_lr:y_lr + lr_patch_size]
    if lr_mask_patch.mean() > useful_region_th:
        lr_image_patch = lr_image[x_lr:x_lr + lr_patch_size, y_lr:y_lr + lr_patch_size]
        return (lr_image_patch, lr_mask_patch, (x_lr, y_lr))
    return None

def generate_patch_pairs(hr_image, hr_mask, lr_image, lr_mask, hr_patch_size=256, lr_patch_size=128, stride=128, scale_factor=2, useful_region_th=0.8):
    """生成 HR 和 LR 的 patch 对，确保两者都满足阈值"""
    patch_pairs = []
    
    # 对 HR 图像进行 Patchify
    hr_patches = patchify_hr(hr_image, hr_mask, hr_patch_size, stride, useful_region_th)
    
    # 遍历每个 HR patch，获取对应的 LR patch
    for hr_patch, hr_mask_patch, hr_coord in hr_patches:
        lr_patch_info = get_lr_patch(lr_image, lr_mask, hr_coord, scale_factor, lr_patch_size, useful_region_th)
        if lr_patch_info is not None:
            lr_patch, lr_mask_patch, lr_coord = lr_patch_info
            patch_pairs.append((hr_patch, hr_mask_patch, hr_coord, lr_patch, lr_mask_patch, lr_coord))
    
    return patch_pairs

def visualize_images(original, padded, title1="Original HR Image", title2="Padded HR Image"):
    """可视化原始图像和扩展后的图像"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap='gray', origin='lower')
    axes[0].set_title(title1)
    axes[1].imshow(padded, cmap='gray', origin='lower')
    axes[1].set_title(title2)
    plt.savefig("/ailab/user/wuguocheng/Astro_SR/vis/vis_padding.png")

def visualize_patch_pair(hr_patch, lr_patch, scale_factor=2):
    """可视化 HR patch 和上采样后的 LR patch"""
    lr_upsampled = zoom(lr_patch, scale_factor, order=1)
    lr_upsampled = lr_upsampled[:hr_patch.shape[0], :hr_patch.shape[1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(hr_patch, cmap='gray')
    axes[0].set_title("HR Patch")
    axes[1].imshow(lr_upsampled, cmap='gray')
    axes[1].set_title("LR Patch (Upsampled)")
    plt.savefig("/ailab/user/wuguocheng/Astro_SR/vis/vis_patch_pair.png")

if __name__ == "__main__":
    # 文件路径
    hr_path = "/ailab/group/pjlab-ai4s/ai4astro/Deep_space_explore/hst_data/hst_13003_39_wfc3_uvis_f390w_ic2i39/hst_13003_39_wfc3_uvis_f390w_ic2i39_drc.fits.gz"
    
    # 加载 HR 图像
    hr_image, hr_mask, hr_wcs = load_fits(hr_path, hr=True)
    
    # Padding HR 图像
    padded_hr_image, padded_hr_wcs = pad_to_multiple(hr_image, hr_wcs, multiple=256, pad_value=np.nan)
    print(f"扩展后 HR 图像尺寸: {padded_hr_image.shape}")
    
    # 下采样生成 LR 图像
    lr_image, lr_wcs = downsample_image(padded_hr_image, padded_hr_wcs, scale_factor=2)
    print(f"lr_image shape: {lr_image.shape}")
    # 保存 lr_image 到一个临时的 FITS 文件
    lr_temp_path = "/ailab/user/wuguocheng/Astro_SR/vis/lr_temp.fits"
    hdu = fits.PrimaryHDU(lr_image, header=lr_wcs.to_header())  # 将图像数据和 WCS 信息保存
    hdu.writeto(lr_temp_path, overwrite=True)  # 写入文件，overwrite=True 允许覆盖已有文件

    # 使用文件路径调用 load_fits
    lr_image, lr_mask, lr_wcs = load_fits(lr_temp_path, hr=False)
    
    # Patchify 参数
    hr_patch_size = 256
    lr_patch_size = hr_patch_size // 2
    stride = 128
    useful_region_th = 0.8
    
    # 生成 HR 和 LR patch 对
    patch_pairs = generate_patch_pairs(padded_hr_image, hr_mask, lr_image, lr_mask, 
                                       hr_patch_size=hr_patch_size, lr_patch_size=lr_patch_size, 
                                       stride=stride, scale_factor=2, useful_region_th=useful_region_th)
    print(f"Patch 对数量: {len(patch_pairs)}")
    
    # 可视化示例 patch 对
    if len(patch_pairs) > 1:
        hr_patch_example, _, _, lr_patch_example, _, _ = patch_pairs[980]
        visualize_patch_pair(hr_patch_example, lr_patch_example)