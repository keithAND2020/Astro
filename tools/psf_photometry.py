from astropy.io import fits
import photutils as phu
import numpy as np
from astropy.wcs import WCS
import tqdm
import warnings
import json
from matplotlib import pyplot as plt
from reproject import reproject_interp, reproject_exact
import pdb
import matplotlib
from matplotlib import pyplot as plt
import astropy.io.fits as pyfits
from astropy.visualization import ZScaleInterval
from matplotlib.colors import Normalize
from PIL import Image
import cv2
from scipy.signal import convolve2d
from scipy.special import j1
from scipy.fft import dct, idct

def fits_vis(ori_array):
    z = ZScaleInterval(n_samples=1000, contrast=0.25)
    z1, z2 = z.get_limits(ori_array)  # 19个一起统计中位数 、 方差
    norm = Normalize(vmin=z1, vmax=z2)
    normalized_array = norm(ori_array)
    cmap = plt.get_cmap('gray')
    wave_array = cmap(normalized_array)
    wave_array = (wave_array[..., 0] * 255).astype(np.uint8)
    return wave_array.astype(np.uint8)


def vis(array):
    try:
        array = fits_vis(array)
    except:
        pass
    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.show()


def perform_psf_photometry(fits_file_path, vis=False):
    from astropy.modeling.fitting import LevMarLSQFitter  # 或者 LinearLSQFitter
    from astropy.io import fits
    import numpy as np
    from scipy.ndimage import zoom
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    from astropy.nddata import NDData
    from astropy.table import Table
    from photutils.psf import EPSFBuilder, extract_stars, BasicPSFPhotometry, DAOGroup

    hdus = fits.open(fits_file_path)
    data = np.nan_to_num(hdus[1].data)

    rows, cols = np.nonzero(data)
    min_row, min_col = min(rows), min(cols)
    max_row, max_col = max(rows), max(cols)
    data = data[min_row:max_row + 1, min_col:max_col + 1]

    scale_factor = (256 / data.shape[0], 256 / data.shape[1])
    data = zoom(data, scale_factor, order=1)
    # scale_factor = (128 / 256, 128 / 256)
    # data = zoom(data, scale_factor, order=1)

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=1.0, threshold=11 * std)  # FWHM可以根据实际情况调整
    sources = daofind(data - median)

    nddata = NDData(data=data)

    stars_tbl = Table()
    stars_tbl['x'] = sources['xcentroid']
    stars_tbl['y'] = sources['ycentroid']

    stars = extract_stars(nddata, stars_tbl, size=5)  # size参数定义了每个星体周围剪裁区域的大小
    epsf_builder = EPSFBuilder(oversampling=2, maxiters=10, progress_bar=True)
    epsf, fitted_stars = epsf_builder(stars)
    basic_fit = BasicPSFPhotometry(group_maker=DAOGroup(0.7),
                                           bkg_estimator=None,
                                           psf_model=epsf,
                                           fitter=LevMarLSQFitter(),
                                           aperture_radius=5.,
                                           fitshape=11,
                                           finder=daofind)

    result_tab = basic_fit(data)

    # 可视化部分开始
    if vis:
        from photutils.aperture import CircularAperture
        import matplotlib.pyplot as plt
        from astropy.visualization import simple_norm
        positions = [(source['xcentroid'], source['ycentroid']) for source in sources]
        apertures = CircularAperture(positions, r=5.)  # 圆形孔径半径可以根据需要调整
        plt.figure(figsize=(10, 10))
        plt.imshow(fits_vis(data), cmap='gray')
        apertures.plot(color='red', lw=1.5, alpha=0.7)

        plt.title('Detected Sources')
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        # plt.savefig("")
        plt.show()
    return sum(result_tab['flux_fit'])


# 假设你有一个路径为 'path/to/your/image.fits' 的FITS文件
fits_file_path = 'hst_skycell-p2433x07y01_acs_wfc_f606w_all_drc.fits'
fluxes = perform_psf_photometry(fits_file_path, vis=True)
print(fluxes)