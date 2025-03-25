import numpy as np
from scipy.ndimage import zoom
from astropy.modeling.models import Gaussian2D, AiryDisk2D
from astropy.convolution import convolve
import matplotlib.pyplot as plt
import pdb
from astropy.visualization import (ZScaleInterval, ImageNormalize)

# 1. 定义模糊核生成函数
def generate_isotropic_gaussian_psf(size, sigma):
    """生成各向同性高斯PSF"""
    half_size = size // 2
    if size % 2 == 0:
        raise ValueError("size must be odd")
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    x, y = np.meshgrid(x, y)
    psf = Gaussian2D(amplitude=1.0, x_mean=0, y_mean=0, x_stddev=sigma, y_stddev=sigma)(x, y)
    return psf / psf.sum()

# def generate_anisotropic_gaussian_psf(size, sigma_x, sigma_y):
#     """生成各向异性高斯PSF"""
#     half_size = size // 2
#     if size % 2 == 0:
#         raise ValueError("size must be odd")
#     x = np.arange(-half_size, half_size + 1)
#     y = np.arange(-half_size, half_size + 1)
#     x, y = np.meshgrid(x, y)
#     psf = Gaussian2D(amplitude=1.0, x_mean=0, y_mean=0, x_stddev=sigma_x, y_stddev=sigma_y)(x, y)
#     return psf / psf.sum()

def generate_airy_psf(size, radius):
    """生成Airy PSF"""
    half_size = size // 2
    if size % 2 == 0:
        raise ValueError("size must be odd")
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    x, y = np.meshgrid(x, y)
    psf = AiryDisk2D(amplitude=1.0, x_0=0, y_0=0, radius=radius)(x, y)
    return psf / psf.sum()

def degrade_patch(image, mask, scale_factor, noise_level_range=[0, 3]):
    psf_type = np.random.choice(['isotropic', 'airy'])
    psf_size = 15 
    if psf_size % 2 == 0:
        raise ValueError("psf_size must be odd")

    # 根据 scale_factor 设置PSF参数范围
    #sigma_range = [0.5, 2]
    if scale_factor == 2:
        sigma_range = [0.2, 2]
    elif scale_factor == 3:
        sigma_range = [0.1, 3]
    elif scale_factor == 4:
        sigma_range = [0.1, 4]
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    if psf_type == 'isotropic':
        psf = generate_isotropic_gaussian_psf(psf_size, sigma)
    # elif psf_type == 'anisotropic':
    #     sigma_x = np.random.uniform(sigma_range[0], sigma_range[1])
    #     sigma_y = np.random.uniform(sigma_range[0], sigma_range[1])
    #     psf = generate_anisotropic_gaussian_psf(psf_size, sigma_x, sigma_y)
    else:  # Airy PSF
        radius = sigma
        psf = generate_airy_psf(psf_size, radius)
    image_temp = np.where(mask, image, 0.0)
    blurred_image = convolve(image_temp, psf, normalize_kernel=True)
    downsample_factor = 1 / scale_factor
    lr_image = zoom(blurred_image, downsample_factor, order=3)
    noise_level = np.random.uniform(noise_level_range[0], noise_level_range[1])
    lr_image += np.random.normal(0, noise_level / 255.0, lr_image.shape)
    lr_image = np.clip(lr_image, 0, None)
    
    return lr_image
if __name__ == '__main__':
    patch_data = np.load('/ailab/user/wuguocheng/AstroIR/tools/creat_dataset/new_create_dataset/train_patches/hst_9453_38_acs_wfc_f814w_j8f838_drc/hst_9453_38_acs_wfc_f814w_j8f838_drc_patch_8.npy', allow_pickle=True).item()
    hr_image = patch_data['image']  
    mask = patch_data['mask']
    scale_factor = 2 
    lr_image = degrade_patch(hr_image,mask, scale_factor)

    norm1 = ImageNormalize(hr_image , interval=ZScaleInterval())
    norm2 = ImageNormalize(lr_image , interval=ZScaleInterval())
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(hr_image, cmap='gray', norm=norm1)
    plt.title('High Resolution Image')
    plt.subplot(1, 2, 2)
    plt.imshow(lr_image, cmap='gray', norm=norm2)
    plt.title('Low Resolution Image')
    plt.savefig('/ailab/user/wuguocheng/AstroIR/tools/creat_dataset/new_create_dataset/vis/vis_hr_lr.png')
    np.save('low_res_patch.npy', lr_image)