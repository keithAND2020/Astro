import torch.nn as nn
import pdb

def weights_init_xavier(m):
    classname = m.__class__.__name__
    try:
        if classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    except:
        pass
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    try:
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    except:
        pass
def weights_init_classifier(m):
    classname = m.__class__.__name__
    try:
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)    
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            try:
                if m.bias:  
                    nn.init.constant_(m.bias, 0.0)
            except:
                nn.init.constant_(m.bias, 0.0)
    except:
        pass