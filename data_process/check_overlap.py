from astropy.io import fits
from astropy.wcs import WCS
from regions import PolygonSkyRegion
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
import regions
from shapely.geometry import Polygon
import os
from tqdm import tqdm
from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation
from scipy.spatial import KDTree
import random
from shapely import wkt


def check(fitsfile1, fitsfile2, history_polygon):
    hdu1 = fits.open(fitsfile1)[1]
    hdu2 = fits.open(fitsfile2)[1]

    wcs1 = WCS(hdu1.header)
    wcs2 = WCS(hdu2.header)

    x = [1, hdu1.data.shape[1], hdu1.data.shape[1], 1]
    y = [1, 1, hdu1.data.shape[0], hdu1.data.shape[0]]
    ra, dec = wcs1.all_pix2world(x, y, 1)
    corners1 = list(zip(ra, dec))

    x = [1, hdu2.data.shape[1], hdu2.data.shape[1], 1]
    y = [1, 1, hdu2.data.shape[0], hdu2.data.shape[0]]
    ra, dec = wcs2.all_pix2world(x, y, 1)
    corners2 = list(zip(ra, dec))

    vertices = SkyCoord([c[0] * u.deg for c in corners1],
                        [c[1] * u.deg for c in corners1], frame='icrs')
    poly_region1 = PolygonSkyRegion(vertices=vertices)

    vertices = SkyCoord([c[0] * u.deg for c in corners2],
                        [c[1] * u.deg for c in corners2], frame='icrs')
    poly_region2 = PolygonSkyRegion(vertices=vertices)

    ra = poly_region1.vertices.ra.deg
    dec = poly_region1.vertices.dec.deg
    region1 = Polygon(np.column_stack((ra, dec)))

    ra = poly_region2.vertices.ra.deg
    dec = poly_region2.vertices.dec.deg
    region2 = Polygon(np.column_stack((ra, dec)))

    intersection = region1.intersection(region2)

    if bool(intersection):
        return True  # 有交集
    else:
        return False  # 没有交集


save_path = 'output_file_path.txt'
path = "/ailab/group/pjlab-ai4s/ai4astro/Deep_space_explore/hst_data/"
directory_paths = []
for root, dirs, files in os.walk(path):
    for dir_name in dirs:
        directory_paths.append(os.path.join(root, dir_name))

with open(save_path, 'a') as f:
    current_path = "/ailab/group/pjlab-ai4s/ai4astro/Deep_space_explore/hst_data/hst_9983_03_acs_wfc_f814w_j8my03/hst_9983_03_acs_wfc_f814w_j8my03_drc.fits.gz"

    ra = np.array([100.96295714, 100.6869888 , 100.68762722, 100.96231893])
    dec = np.array([-74.2752937 , -74.27529367, -74.20030779, -74.20030782])
    region = Polygon(np.column_stack((ra, dec)))
    polygon_wkt = region.wkt
    f.write(f"{current_path}:{polygon_wkt}\n")


for fits_dir_path in tqdm(directory_paths):
    try:
        root_dir, _, current_files = next(os.walk(fits_dir_path))
        matching_files = [file for file in current_files if "drc" in file]
        fits_filepath = os.path.join(root_dir, matching_files[0])
        hdu1 = fits.open(fits_filepath)[1]
        wcs1 = WCS(hdu1.header)
        x = [1, hdu1.data.shape[1], hdu1.data.shape[1], 1]
        y = [1, 1, hdu1.data.shape[0], hdu1.data.shape[0]]
        ra, dec = wcs1.all_pix2world(x, y, 1)
        corners1 = list(zip(ra, dec))
        vertices = SkyCoord([c[0] * u.deg for c in corners1],
                            [c[1] * u.deg for c in corners1], frame='icrs')
        poly_region1 = PolygonSkyRegion(vertices=vertices)
        ra = poly_region1.vertices.ra.deg
        dec = poly_region1.vertices.dec.deg
        region1 = Polygon(np.column_stack((ra, dec)))

        with open(save_path, 'r+') as f:
            lines = f.readlines()
            lines = [i.strip() for i in lines]
            tot = 0
            for history_path in lines:
                polygon = history_path.split(":")[-1]
                history_polygon = wkt.loads(polygon)
                intersection = region1.intersection(history_polygon)
                if bool(intersection):
                    tot += 1
                    break
            if tot == 0:
                f.write(f"{fits_filepath}:{region1.wkt}\n")
    except:
        pass

print("done!")

