import os
import random
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve
from astropy.modeling.functional_models import Gaussian2D, AiryDisk2D
from reproject import reproject_exact
import numpy as np
from shapely.wkt import loads
import pdb
# 生成随机 PSF
def generate_random_psf(size=15, sigma_range=[0.8, 2.8], radius_range=[1.5, 3.0]):
    """生成随机 PSF，从 Gaussian 和 Airy 中随机选择，并打印参数"""
    psf_type = random.choice(['gaussian', 'airy'])
    if psf_type == 'gaussian':
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        half_size = size // 2
        x = np.arange(-half_size, half_size + 1)
        y = np.arange(-half_size, half_size + 1)
        x, y = np.meshgrid(x, y)
        psf = Gaussian2D(amplitude=1.0, x_mean=0, y_mean=0, x_stddev=sigma, y_stddev=sigma)(x, y)
        print(f"Generated Gaussian PSF with sigma = {sigma}")
    elif psf_type == 'airy':
        radius = random.uniform(radius_range[0], radius_range[1])
        half_size = size // 2
        x = np.arange(-half_size, half_size + 1)
        y = np.arange(-half_size, half_size + 1)
        x, y = np.meshgrid(x, y)
        psf = AiryDisk2D(amplitude=1.0, x_0=0, y_0=0, radius=radius)(x, y)
        print(f"Generated Airy PSF with radius = {radius}")
    return psf / psf.sum()

def load_fits(file_path):
    """加载 FITS 文件，返回图像数据、掩码和 WCS 信息"""
    with fits.open(file_path) as hdul:
        image = hdul[1].data.astype(float)
        wcs = WCS(hdul[1].header)
        mask = ~np.isnan(image)
        if np.all(mask):
            mask = np.ones_like(image, dtype=bool)
        if image.shape[0] > 6000 or image.shape[1] > 6000:
            raise ValueError(f"图像 shape {image.shape} 过大，跳过处理")
        return image, mask, wcs

def apply_psf(image, psf, mask=None):
    """应用 PSF 模糊，mask 为 None 时使用图像默认掩码"""
    if mask is None:
        mask = ~np.isnan(image)
    image_temp = np.where(mask, image, 0.0)
    blurred_image = convolve(image_temp, psf, normalize_kernel=True)
    blurred_image[~mask] = np.nan
    return blurred_image

def pad_to_multiple(image, mask, wcs, multiple=256, pad_value=np.nan):
    """将图像和掩码 padding 到指定倍数，并更新 WCS"""
    h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple  
    pad_w = (multiple - w % multiple) % multiple  
    
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_value)
    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)
    
    padded_wcs = wcs.deepcopy()
    padded_wcs.array_shape = padded_image.shape
    
    return padded_image, padded_mask, padded_wcs

def downsample_image(image, wcs, scale_factor=2):
    """下采样图像并更新 WCS"""
    target_shape = (int(image.shape[0] / scale_factor), int(image.shape[1] / scale_factor))
    target_wcs = wcs.deepcopy()
    target_wcs.wcs.crpix = [crpix / scale_factor for crpix in wcs.wcs.crpix]
    if hasattr(wcs.wcs, 'cd'):
        target_wcs.wcs.cd = wcs.wcs.cd * scale_factor
    else:
        target_wcs.wcs.cdelt = [cdelt * scale_factor for cdelt in wcs.wcs.cdelt]
    downsampled_image, _ = reproject_exact((image, wcs), target_wcs, shape_out=target_shape)
    return downsampled_image, target_wcs

def save_downsampled_image(image, wcs, output_dir, identifier):
    """保存下采样后的图像为 FITS 文件"""
    lr_path = os.path.join(output_dir, f"{identifier}_downsampled.fits")
    hdu = fits.PrimaryHDU(image, header=wcs.to_header())
    hdu.writeto(lr_path, overwrite=True)
    return lr_path

def process_fits_files(datasetlist_path, output_dir, split_file_dir, scale_factor=2):
    """处理 FITS 文件，生成训练和验证集"""
    os.makedirs(split_file_dir, exist_ok=True)
    train_files_path = os.path.join(split_file_dir, "train_files.txt")
    eval_files_path = os.path.join(split_file_dir, "eval_files.txt")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(train_files_path) and os.path.exists(eval_files_path):
        print("Detect existing train_files.txt and eval_files.txt and read them directly...")
        with open(train_files_path, "r") as f:
            train_files = [line.strip().split(',') for line in f.readlines()]
        with open(eval_files_path, "r") as f:
            eval_files = [line.strip().split(',') for line in f.readlines()]
    else:
        with open(datasetlist_path, "r") as f:
            lines = f.readlines()
        train_files = []
        eval_files = []
        for line in tqdm(lines, desc="Filtering and partitioning datasets"):
            try:
                fits_filepath, wkt_str = line.strip().split(":")
                with fits.open(fits_filepath) as hdul:
                    header = hdul[1].header
                    ncombine = header.get('NCOMBINE', 0)
                    if ncombine == 4:
                        polygon = loads(wkt_str)
                        ra_values = [point[0] for point in polygon.exterior.coords]
                        min_ra = min(ra_values)
                        max_ra = max(ra_values)
                        image, mask, wcs = load_fits(fits_filepath)
                        # Padding 图像和掩码
                        padded_image, padded_mask, padded_wcs = pad_to_multiple(image, mask, wcs, multiple=256, pad_value=np.nan)
                        if max_ra < 250:
                            train_files.append((fits_filepath, padded_image, padded_mask, padded_wcs))
                        elif min_ra > 255:
                            eval_files.append((fits_filepath, padded_image, padded_mask, padded_wcs))
            except ValueError as ve:
                print(f"skip {fits_filepath}: {ve}")
            except Exception as e:
                print(f"skip {fits_filepath} fail: {e}")

        with open(train_files_path, "w") as f_train, open(eval_files_path, "w") as f_eval:
            for fits_filepath, padded_image, padded_mask, padded_wcs in tqdm(train_files, desc="Processing train files"):
                try:
                    psf = generate_random_psf()
                    blurred_image = apply_psf(padded_image, psf, mask=padded_mask)
                    downsampled_image, target_wcs = downsample_image(blurred_image, padded_wcs, scale_factor)
                    identifier = os.path.basename(fits_filepath).replace(".fits", "").replace(".gz", "")
                    lr_path = save_downsampled_image(downsampled_image, target_wcs, output_dir, identifier)
                    f_train.write(f"{fits_filepath},{lr_path}\n")
                except Exception as e:
                    print(f"Processing {fits_filepath} fail: {e}")

            for fits_filepath, padded_image, padded_mask, padded_wcs in tqdm(eval_files, desc="Processing validation set files"):
                try:
                    psf = generate_random_psf()
                    blurred_image = apply_psf(padded_image, psf, mask=padded_mask)
                    downsampled_image, target_wcs = downsample_image(blurred_image, padded_wcs, scale_factor)
                    identifier = os.path.basename(fits_filepath).replace(".fits", "").replace(".gz", "")
                    lr_path = save_downsampled_image(downsampled_image, target_wcs, output_dir, identifier)
                    f_eval.write(f"{fits_filepath},{lr_path}\n")
                except Exception as e:
                    print(f"Processing verification documents {fits_filepath} fail: {e}")

if __name__ == "__main__":
    datasetlist_path = "/ailab/user/wuguocheng/Astro_SR/data_process/split_file/datasetlist.txt"
    output_dir = "/ailab/user/wuguocheng/Astro_SR/dataset/psf_downsampled"
    split_file_dir = "/ailab/user/wuguocheng/Astro_SR/data_process/split_file"
    process_fits_files(datasetlist_path, output_dir, split_file_dir)