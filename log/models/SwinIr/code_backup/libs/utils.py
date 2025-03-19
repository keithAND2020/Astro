import torch
import numpy as np
import random
import os
import sys
import pdb
import logging

import math
from typing import Optional, Tuple
from importlib import import_module
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
from io import BytesIO

import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from multiprocessing import Pool
import subprocess

import time as time

import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np

from torch import default_generator, randperm



def init_logger(config):
    log_file = os.path.join(config['train']['log_dir'],'train.log')
    rank = config['train']['local_rank']
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO if rank == 0 else 'ERROR',
                        format=log_format,
                        filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else 'ERROR')
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def init_config(args):

    def fuse_config(base_cfg, cfg):
        for key in cfg.keys():
            if type(cfg[key]) is dict and type(base_cfg[key]) is dict:
                base_cfg[key] = fuse_config(base_cfg[key], cfg[key])
            else:
                if cfg.get('_delete_',False):
                    cfg.pop('_delete_')
                    return cfg
                else:
                    base_cfg[key] = cfg[key]
        return base_cfg
    def load_config(root_dir, cfg_name):
        sys.path.insert(0, root_dir)

        pyconfigs = import_module(cfg_name)
        configs = {}
        sys.path.pop(0)
        for key in dir(pyconfigs):
            if '__'!=key[:2]:
                component = getattr(pyconfigs,key)
                if type(component)==tuple: component = component[0]
                configs[key] = component
        if '_base_' in configs:
            base_infos = configs.pop('_base_').split('/')
            base_dir = "".join(base_infos[:-1])
            base_name = "".join(base_infos[-1:])
            base_root = os.path.join(root_dir,base_dir)
            base_config = load_config(base_root,base_name[:-3])
            configs = fuse_config(base_config, configs)
        return configs
    root_dir = os.path.abspath(os.path.dirname(args.config))
    cfg_name = args.config.split('/')[-1][:-3]
    
    configs = load_config(root_dir,cfg_name)

    if args.log_dir is not None:
        configs['train']['log_dir']=os.path.join('./log',args.log_dir)
    else:
        configs['train']['log_dir']=os.path.join('./log',args.config.split('configs')[-1][1:-3])
    os.makedirs(configs['train']['log_dir'],exist_ok=True)
    # configs['test']['visualize'] = args.visualize
    return configs

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)
    torch.backends.cudnn.enabled = False
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def backup_code(backup_lists, target_dir):
    target_dir = os.path.join(target_dir,'code_backup')
    os.makedirs(target_dir,exist_ok=True)
    for _ in backup_lists:
        source_dir = os.path.join(os.getcwd(),_)
        os.system('cp -r '+source_dir+' '+target_dir)
        print('Have backuped '+
              '\033[0;32;40mcontent\033[0m'.replace('content',_)+
              ' to '+
              '\033[0;35;40mcontent\033[0m'.replace('content',target_dir))

def init_ddp(args,configs): 
    configs['train']['local_rank'] = 0
    # configs['test']['local_rank'] = 0
    if args.launcher != 'none':
        _, configs['dataset']['local_rank'], configs['dataset']['world_size'] = init_dist_slurm(backend='nccl')
        # _, configs['dataset']['local_rank'], configs['dataset']['world_size'] = init_dist_pytorch(args.tcp_port, backend='nccl')
        configs['dataset']['ddp'] = True
        configs['train']['local_rank'] = configs['dataset']['local_rank']
        # configs['test']['local_rank'] = configs['dataset']['local_rank']
        configs['train']['ddp'] = True
        # configs['test']['ddp'] = True
    return configs

def init_all(args):
    configs = init_config(args)
    torch.set_default_dtype(getattr(torch,configs['dataset'].get('data_type','float32')))
    configs = init_ddp(args,configs)
    init_seed(configs['seed']+configs['train']['local_rank'])
    logger = init_logger(configs)
    logger.info('Running setting: {}'.format(args.config))
    if configs['train']['local_rank']==0:
        backup_lists = ['libs','tools','configs']
        backup_code(backup_lists, configs['train']['log_dir'])
    configs['model']['checkpoint'] = args.resume
    return configs, logger


def init_dist_slurm(tcp_port=None, backend='nccl'):
    num_gpus = torch.cuda.device_count()
    rank = int(os.environ["RANK"])
    # world_size = int(os.environ["SLURM_NTASKS"])
    world_size = int(os.environ["WORLD_SIZE"])
    # node_list = os.environ["SLURM_NODELIST"]
    # addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if "MASTER_PORT" not in os.environ and tcp_port is not None:
        print("MASTER_PORT not in os.environ")
        os.environ["MASTER_PORT"] = str(tcp_port)
    elif "MASTER_PORT" not in os.environ:
        print("MASTER_PORT not in os.environ")
        os.environ["MASTER_PORT"] = "29501"

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr
    # os.environ["WORLD_SIZE"] = str(world_size)
    # os.environ["LOCAL_RANK"] = str(rank % num_gpus)
    # os.environ["RANK"] = str(rank)

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return num_gpus, rank, world_size


def init_dist_pytorch(tcp_port, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    rank, world_size = get_dist_info()
    return num_gpus, rank, world_size

def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


