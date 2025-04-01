from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_exact
import numpy as np
import pdb
with fits.open('/ailab/group/pjlab-ai4s/ai4astro/Deep_space_explore/hst_data/jd8f28020/jd8f28020_drc.fits.gz') as hdul:
    data = hdul[1].data
    header = hdul[1].header
    wcs_original = WCS(header)
ny, nx = data.shape  # (4218, 4244)
shape_target = (ny // 2, nx // 2)  # (2109, 2122)
wcs_target = wcs_original.deepcopy()
if hasattr(wcs_original.wcs, 'cd'):
    wcs_target.wcs.cd = wcs_original.wcs.cd * 2  # 像素尺度放大2倍
else:
    wcs_target.wcs.cdelt = wcs_original.wcs.cdelt * 2  # 否则调整 cdelt
wcs_target.wcs.crpix /= 2
wcs_target.pixel_shape = shape_target
data_reprojected, footprint = reproject_exact((data, wcs_original), wcs_target, shape_out=shape_target)
if np.all(footprint == 0):
    print("错误：目标图像没有覆盖原始数据！")
else:
    print("下采样成功！")
    # 计算通量
    flux_original = np.nansum(data)
    flux_reprojected = np.nansum(data_reprojected)
    print(f"原始图像总通量: {flux_original}")
    print(f"下采样后图像总通量: {flux_reprojected}")
    print(f"通量差异: {flux_original - flux_reprojected}")
    # 保存结果
    hdu = fits.PrimaryHDU(data=data_reprojected, header=wcs_target.to_header())
    hdu.writeto('downsampled_image.fits', overwrite=True)