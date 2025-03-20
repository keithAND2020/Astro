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
        for datalist in self.evalloader:  
            infer_datalist = datalist.copy()
            for key in infer_datalist.keys():
                if type(infer_datalist[key]) is torch.Tensor:
                    infer_datalist[key] = infer_datalist[key].to('cuda')
            with torch.no_grad():
                results = self.model(infer_datalist['input'], infer_datalist)
                pdb.set_trace()
                results = {key:results[key].cpu() if type(results[key]) is torch.Tensor else results[key] for key in results.keys()}
            #eval_string, eval_dict = evaluate_metric(results, datalist, self.ddp)
            #if self.visualize: vis_results(results, datalist, self.vis_dir)

        
