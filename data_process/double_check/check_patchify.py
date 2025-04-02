import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from tqdm import tqdm

# 设置 Z-scale 归一化函数
def z_scale_image(image):
    """对图像应用 Z-scale 归一化"""
    z = ZScaleInterval()
    vmin, vmax = z.get_limits(image)
    normalized_image = np.clip((image - vmin) / (vmax - vmin), 0, 1)  # 归一化到 [0, 1]
    return normalized_image

# 可视化 HR 和 LR patch 对
def visualize_patch_pair(hr_patch, lr_patch, identifier, idx, output_dir):
    """可视化 HR 和 LR patch 对，并保存到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 应用 Z-scale 归一化
    hr_patch_normalized = z_scale_image(hr_patch)
    lr_patch_normalized = z_scale_image(lr_patch)
    
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示 HR patch
    axes[0].imshow(hr_patch_normalized, cmap='gray')
    axes[0].set_title(f"HR Patch {idx} ({hr_patch.shape[0]}x{hr_patch.shape[1]})")
    axes[0].axis('off')
    
    # 显示 LR patch
    axes[1].imshow(lr_patch_normalized, cmap='gray')
    axes[1].set_title(f"LR Patch {idx} ({lr_patch.shape[0]}x{lr_patch.shape[1]})")
    axes[1].axis('off')
    
    # 保存图像
    output_path = os.path.join(output_dir, f"{identifier}_patch_{idx}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 主函数：加载并可视化所有 patch
def visualize_all_patches(hr_patch_dir, lr_patch_dir, output_dir, identifier):
    """加载并可视化所有 HR 和 LR patch 对"""
    # 获取 HR 和 LR patch 文件列表
    hr_files = sorted([f for f in os.listdir(hr_patch_dir) if f.startswith(identifier) and f.endswith('.npy')])
    lr_files = sorted([f for f in os.listdir(lr_patch_dir) if f.startswith(identifier) and f.endswith('.npy')])
    
    # 检查文件数量是否一致
    if len(hr_files) != len(lr_files):
        print(f"警告: HR 文件数量 ({len(hr_files)}) 与 LR 文件数量 ({len(lr_files)}) 不一致")
    
    # 遍历所有 patch 对
    for idx, (hr_file, lr_file) in enumerate(tqdm(zip(hr_files, lr_files), total=min(len(hr_files), len(lr_files)), desc="Visualizing patches")):
        hr_data = np.load(os.path.join(hr_patch_dir, hr_file), allow_pickle=True).item()
        lr_data = np.load(os.path.join(lr_patch_dir, lr_file), allow_pickle=True).item()
        
        hr_patch = hr_data['image']  # 提取 HR patch 的图像数据
        lr_patch = lr_data['image']  # 提取 LR patch 的图像数据
        
        # 可视化并保存
        visualize_patch_pair(hr_patch, lr_patch, identifier, idx, output_dir)

if __name__ == "__main__":
    # 设置目录和样本标识符
    hr_patch_dir = "/ailab/user/wuguocheng/Astro_SR/dataset/train_hr_patch"
    lr_patch_dir = "/ailab/user/wuguocheng/Astro_SR/dataset/train_lr_patch"
    output_dir = "/ailab/user/wuguocheng/Astro_SR/vis/temp_npy"
    
    # 假设样本的 identifier，例如从文件名中提取
    identifier = "jd8f28020_drc"  # 根据您的样本文件名调整
    
    # 运行可视化
    visualize_all_patches(hr_patch_dir, lr_patch_dir, output_dir, identifier)