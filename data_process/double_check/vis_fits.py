import json
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import pdb
with open("train.json", "r") as f:
    train_files = json.load(f)

fits_filepath = train_files[4]

hdu = fits.open(fits_filepath)[1]
pdb.set_trace()
print(fits.info(fits_filepath))
data = hdu.data
norm = ImageNormalize(data, interval=ZScaleInterval())

plt.figure(figsize=(16, 16))

plt.imshow(data, cmap='gray', origin='lower', norm=norm)
plt.colorbar(label='Intensity')
plt.title(f"FITS Image: {fits_filepath}")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.savefig('/ailab/user/wuguocheng/AstroIR/tools/creat_dataset/new_create_dataset/vis/result_fits.png')