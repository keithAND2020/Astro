import torch
import torch.nn as nn
import pdb
from .model_init import *
import warnings

class Base_Model(nn.Module):
    def __init__(self, 
                 losses,
                 initializer=None,
                 **kargs):
        super(Base_Model, self).__init__()
        from ..builders import build_losses
        for loss in losses.keys():
            setattr(self, loss, build_losses(**losses[loss]))
        self.initializer = initializer
        if len(kargs.keys())>0:
            warnings.warn("keys:{} are not used".format(kargs.keys()))
        
    def init_weights(self):
        if self.initializer is not None and self.initializer != 'gaussian':
            self.apply(globals()['weights_init_'+self.initializer])        
