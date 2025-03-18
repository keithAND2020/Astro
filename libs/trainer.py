import pdb
import os
import torch
import sys
import time
import tqdm
from matplotlib.colors import Normalize
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import glob
import random
import re
import itertools
from itertools import islice
import torchvision.transforms as transforms
from einops import repeat, rearrange
from astropy.visualization import ZScaleInterval
from matplotlib.colors import Normalize
from PIL import Image
import cv2
from .photometric_loss import PhotometricLoss
from pytorch_msssim import ssim
import piqa

class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 trainloader,
                 evalloader,
                 logger,
                 max_epoch,
                 log_dir,
                 grad_clip=None,
                 local_rank=0,
                 ddp=False,
                 save_ckp_epoch=10,
                 eval_epoch=10,
                 display_iter=5):
        self.model = model
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epoch = max_epoch
        self.trainloader = trainloader
        self.evalloader = evalloader
        self.log_dir = log_dir
        self.save_ckp_epoch = save_ckp_epoch
        self.display_iter = display_iter
        self.eval_epoch = eval_epoch
        self.epoch = 0
        self.grad_clip = grad_clip
        self.ddp = ddp
        self.rank = local_rank
        if self.ddp:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[local_rank % torch.cuda.device_count()],
                                                             find_unused_parameters=False)


    def train(self):
        criterion = torch.nn.L1Loss()
        criterion = torch.nn.MSELoss()
        # criterion = PhotometricLoss()

        with tqdm.tqdm(range(self.max_epoch), desc='Epoch:', smoothing=1) as pbar:
            min_loss = 1e9
            for epoch in pbar:
                self.epoch = epoch + 1
                if self.ddp:
                    dist.barrier()
                    self.trainloader.sampler.set_epoch(self.epoch)
                    total_losses = []

                for idx, (origin, gt, (meann, stdd)) in enumerate(self.trainloader):
                    origin = origin.to("cuda")
                    gt = gt.to("cuda")
                    pred = self.model(origin)

                    loss = criterion(pred, gt)
                    self.optimizer.zero_grad()
                    loss.backward()

                    self.optimizer.step()
                    total_losses.append(loss.item())

                self.scheduler.step()
                avg_loss = sum(total_losses) / len(total_losses)

                if avg_loss < min_loss and self.rank == 0:
                    min_loss = avg_loss
                    torch.save(self.model.module.state_dict(), os.path.join(self.log_dir, f'best.pth'))

                pbar.set_postfix(lr=self.optimizer.param_groups[0]['lr'], MSELoss=avg_loss)

        pbar.close()


    def train_one_epoch(self):
        self.model.train()
        for i, datalist in enumerate(self.trainloader):
            for key in datalist.keys():
                datalist[key] = datalist[key].to("cuda")
            inp = datalist['inputs']
            tar = datalist['targets']
            self.optimizer.zero_grad()
            pred = self.model(inp)
            loss = self.criterion(pred, tar)
            
    def eval(self, save_path="/ailab/user/liziyang/workspace/AstroIR/figs/", vis=False):
        model_name = self.log_dir
        self.model.module.load_state_dict(torch.load(f"{model_name}/best.pth"))
        criterion = PhotometricLoss()
        device = "cuda"
        PSNR_list = []
        SSIM_list = []
        TFE_list = []
        with torch.no_grad():
            # batch = 1
            for idx, (origin, gt, (meann, stdd)) in enumerate(tqdm.tqdm(self.evalloader, desc="Evaluating")):
                # origin = origin.to(device)

                pred = origin.clone()
                pred = pred * stdd + meann

                # pred = self.model(origin)
                # pred = pred.to("cpu") * stdd + meann
                gt = gt * stdd + meann

                TFE_list.append(criterion(pred, gt))
                PSNR_list.append(self.calculate_psnr(pred, gt))
                SSIM_list.append(self.calculate_ssim(pred, gt))

                tqdm.tqdm.write(f"Batch {idx}: PSNR={PSNR_list[idx]:.4f}, SSIM={SSIM_list[idx]:.4f}, TFE={TFE_list[idx]:.4f}")

                if vis:
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                    axes[1].imshow(gt[0, 0, :, :], cmap='gray')
                    axes[1].axis('off')  # Hide the axes
                    axes[1].set_title('Ground Truth')
                    axes[0].imshow(pred[0, 0, :, :], cmap='gray')
                    axes[0].axis('off')  # Hide the axes
                    axes[0].set_title('Prediction')
                    plt.tight_layout()
                    plt.savefig(save_path + f'{model_name.split("/")[-1]}{idx}.png', dpi=600, bbox_inches='tight')
                    plt.show()

        print(model_name, "---- PSNR:",np.mean(PSNR_list),  "    SSIM:", np.mean(SSIM_list),  "    TFE:", np.nanmean(TFE_list))

        sys.exit(0)

    def fits_vis(self, ori_array):
        z = ZScaleInterval(n_samples=1000, contrast=0.25)
        z1, z2 = z.get_limits(ori_array)  # 19个一起统计中位数 、 方差
        norm = Normalize(vmin=z1, vmax=z2)
        normalized_array = norm(ori_array)
        cmap = plt.get_cmap('gray')
        wave_array = cmap(normalized_array)
        wave_array = (wave_array[..., 0] * 255).astype(np.uint8)
        return wave_array.astype(np.uint8)

    def calculate_psnr(self, img1, img2):
        psnr_calculator = piqa.psnr.PSNR()
        img1 = img1.expand(-1, 3, -1, -1)
        img2 = img2.expand(-1, 3, -1, -1)

        img_min = img1.min()
        img_max = img1.max()
        img1 = (img1 - img_min) / (img_max - img_min)
        img_min = img2.min()
        img_max = img2.max()
        img2 = (img2 - img_min) / (img_max - img_min)

        psnr_value = psnr_calculator(img1, img2)
        return psnr_value


    def calculate_ssim(self, img1, img2, window_size=11, sigma=1.5):
        img_min = img1.min()
        img_max = img1.max()
        img1_normalized = (img1 - img_min) / (img_max - img_min)

        img_min = img2.min()
        img_max = img2.max()
        img2_normalized = (img2 - img_min) / (img_max - img_min)

        # Calculate and return the SSIM value
        return ssim(img1_normalized, img2_normalized, data_range=1.0, win_size=window_size, win_sigma=sigma).item()


    def remove_module_from_state_dict(self, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 移除 `module.`
            new_state_dict[name] = v
        return new_state_dict