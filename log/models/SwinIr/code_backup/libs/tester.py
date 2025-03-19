import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde, linregress
from tqdm import tqdm
import torch.distributed as dist


class Tester(object):
    def __init__(self, 
                 model, 
                 evalloader, 
                 local_rank=0,
                 ddp=False,
                 visualize=False,
                 vis_dir=None):
        self.model = model
        self.evalloader = evalloader
        self.visualize = visualize
        self.vis_dir = vis_dir
        self.ddp = ddp
        self.local_rank = local_rank
        if ddp and type(self.model) is not nn.parallel.DistributedDataParallel:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[local_rank % torch.cuda.device_count()],
                                                             find_unused_parameters=False)  
        if self.vis_dir is not None:
            os.makedirs(vis_dir,exist_ok=True)

    def eval(self):
        self.model.eval()
        
