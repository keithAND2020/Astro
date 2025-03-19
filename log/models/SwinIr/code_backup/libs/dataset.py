import os
import pdb
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings('ignore', category=UserWarning)

class SR_dataset(Dataset):
    def __init__(self, split, root_dir, **kargs):
        self.root_dir = root_dir
        self.split = split
        self.mode = kargs.get('mode', 'default') 

        if split == 'train':
            with open(kargs['filenames_file_train'], 'r') as f:
                self.filenames = [line.strip() for line in f.readlines() if line.startswith('train')]
        elif split == 'eval':
            with open(kargs.get['filenames_file_eval'], 'r') as f:
                self.filenames = [line.strip() for line in f.readlines() if line.startswith('eval')]
        else:
            raise ValueError("split must be 'train' or 'eval'")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        line = self.filenames[index]
        _, hr_file_path, lr_file_path = line.split(',')
        pdb.set_trace()
        hr_data = np.load(hr_file_path, allow_pickle=True).item()
        hr_image = hr_data['image'] 
        mask = hr_data['mask']
        lr_image = np.load(lr_file_path)
        #!  temp

        from astropy.visualization import (ZScaleInterval, ImageNormalize)
        import matplotlib.pyplot as plt
        norm1 = ImageNormalize(hr_image, interval=ZScaleInterval())
        norm2 = ImageNormalize(lr_image, interval=ZScaleInterval())
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(hr_image, cmap='gray', norm=norm1)
        plt.title('High Resolution Image')
        plt.subplot(1, 2, 2)
        plt.imshow(lr_image, cmap='gray', norm=norm2)
        plt.title('Low Resolution Image')
        plt.savefig('/ailab/user/wuguocheng/Astro_SR/vis/vis_hr_lr.png')
        pdb.set_trace()        


        hr_image = self.normalize(hr_image)
        lr_image = self.normalize(lr_image)



        mask = torch.from_numpy(mask).float()
        return {'lr': torch.from_numpy(lr_image).float(), 
                'hr': torch.from_numpy(hr_image).float(), 
                'mask': mask}

    def normalize(self, image):
        return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
