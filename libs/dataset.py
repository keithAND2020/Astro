import os
import pdb
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset
import random  
warnings.filterwarnings('ignore', category=UserWarning)

class SR_dataset(Dataset):
    def __init__(self, split, root_dir, **kargs):
        self.root_dir = root_dir
        self.split = split
        self.mode = kargs.get('mode', 'default') 
        seed = 42  # 固定的随机种子  
        random.seed(seed)  # 设置随机种子  

        if split == 'train':
            with open(kargs['filenames_file_train'], 'r') as f:
                self.filenames = [line.strip() for line in f.readlines()]# if line.startswith('train')]
                random.shuffle(self.filenames)  
                split_index = int(0.8 * len(self.filenames))  
                self.filenames = self.filenames[:split_index]  
        elif split == 'eval':
            with open(kargs['filenames_file_eval'], 'r') as f:
                self.filenames = [line.strip() for line in f.readlines()]# if line.startswith('eval')]
                random.shuffle(self.filenames)  
                split_index = int(0.8 * len(self.filenames))  
                self.filenames = self.filenames[split_index:]  
        else:
            raise ValueError("split must be 'train' or 'eval'")

    def __len__(self):
        return len(self.filenames)


    # def __getitem__(self, index):  
    #     line = self.filenames[index]  
    #     hr_file_path, lr_file_path, _, _ = line.split(',')  

    #     # 加载高分辨率（HR）图像和掩码  
    #     hr_data = np.load(hr_file_path, allow_pickle=True).item()  
    #     hr_image = hr_data['image']   # 形状： (256, 256)  
    #     mask = hr_data['mask']         # 形状： (256, 256)  

    #     # 加载低分辨率（LR）图像和掩码  
    #     lr_data = np.load(lr_file_path, allow_pickle=True).item()  
    #     lr_image = lr_data['image']    # 形状： (128, 128)  
    #     lr_mask = lr_data['mask']       # 形状： (128, 128)  

    #     # 正常化图像  
    #     hr_image = self.normalize(hr_image, mask)  # 正常化 HR 图像  
    #     lr_image = self.normalize(lr_image, lr_mask)  # 正常化 LR 图像  

    #     # 将NumPy数组转换为PyTorch张量  
    #     hr_image = torch.from_numpy(hr_image).float()  
    #     lr_image = torch.from_numpy(lr_image).float()  
    #     mask = torch.from_numpy(mask).float()  
    #     lr_mask = torch.from_numpy(lr_mask).float()  

    #     # 随机决定翻转的方式：不翻转、水平翻转、垂直翻转、转置  
    #     flip_option = np.random.choice(['none', 'horizontal', 'vertical', 'transpose'])  

    #     if flip_option == 'horizontal':  
    #         hr_image = torch.flip(hr_image, dims=[1])  # 横向翻转  
    #         lr_image = torch.flip(lr_image, dims=[1])  # 横向翻转  
    #         mask = torch.flip(mask, dims=[1])          # 横向翻转  
    #         lr_mask = torch.flip(lr_mask, dims=[1])    # 横向翻转  
    #     elif flip_option == 'vertical':  
    #         hr_image = torch.flip(hr_image, dims=[0])  # 纵向翻转  
    #         lr_image = torch.flip(lr_image, dims=[0])  # 纵向翻转  
    #         mask = torch.flip(mask, dims=[0])          # 纵向翻转  
    #         lr_mask = torch.flip(lr_mask, dims=[0])    # 纵向翻转  

    #     # 扩展维度  
    #     hr_image = hr_image.unsqueeze(0)  # 扩展 HR 图像维度到 (1, 256, 256)  
    #     lr_image = lr_image.unsqueeze(0)  # 扩展 LR 图像维度到 (1, 128, 128)  
    #     mask = mask.unsqueeze(0)          # 扩展 HR 掩码维度到 (1, 256, 256)  
    #     lr_mask = lr_mask.unsqueeze(0)    # 扩展 LR 掩码维度到 (1, 128, 128)  

    #     return {  
    #         'input': lr_image,           # LR 图像张量  
    #         'hr': hr_image,              # HR 图像张量  
    #         'mask': mask,                # HR 掩码张量  
    #         'lr_mask': lr_mask,          # LR 掩码张量  
    #         'filename': hr_file_path.split('/')[-1],  # 文件名  
    #         'item': index                # 索引  
    #     }
    def __getitem__(self, index):
        line = self.filenames[index]
        hr_file_path, lr_file_path ,_,_= line.split(',')
        
        hr_data = np.load(hr_file_path, allow_pickle=True).item()
        hr_image = hr_data['image'] 
        mask = hr_data['mask']


        lr_data = np.load(lr_file_path, allow_pickle=True).item()
        lr_image = lr_data['image'] 
        lr_mask = lr_data['mask']

        ''' temp for vis
        # from astropy.visualization import (ZScaleInterval, ImageNormalize)
        # import matplotlib.pyplot as plt
        # norm1 = ImageNormalize(hr_image, interval=ZScaleInterval())
        # norm2 = ImageNormalize(lr_image, interval=ZScaleInterval())
        # plt.figure(figsize=(15, 10))
        # plt.subplot(1, 2, 1)
        # plt.imshow(hr_image, cmap='gray', norm=norm1)
        # plt.title('High Resolution Image')
        # plt.subplot(1, 2, 2)
        # plt.imshow(lr_image, cmap='gray', norm=norm2)
        # plt.title('Low Resolution Image')
        # plt.savefig('/ailab/user/wuguocheng/Astro_SR/vis/vis_hr_lr.png')
        # pdb.set_trace()   
        '''
        hr_image = self.normalize(hr_image, mask)
        lr_image = self.normalize(lr_image,lr_mask)
        
        hr_image = np.expand_dims(hr_image, axis=0)
        lr_image = np.expand_dims(lr_image, axis=0)
        mask = torch.from_numpy(mask).float()
        mask = np.expand_dims(mask, axis=0)
        return {'input': torch.from_numpy(lr_image).float(), 
                'hr': torch.from_numpy(hr_image).float(), 
                'mask': mask,
                'filename': hr_file_path.split('/')[-1],
                'item': index}

    def normalize(self, image, mask=None):
        if mask is not None:
            valid_pixels = image[mask]
            if len(valid_pixels) > 0:
                min_val = np.min(valid_pixels)
                max_val = np.max(valid_pixels)
            else:
                min_val = 0
                max_val = 1
            image_normalized = (image - min_val) / (max_val - min_val + 1e-8)
            image_normalized[~mask] = 0  
        else:
            min_val = np.min(image)
            max_val = np.max(image)
            image_normalized = (image - min_val) / (max_val - min_val + 1e-8)
        return image_normalized
