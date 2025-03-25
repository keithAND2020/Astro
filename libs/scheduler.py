import numpy as np
import pdb

def linear_warmup(iter, bound):
    return iter/bound if iter<bound else 1.0

class cos_lrdecay(object):
    def __init__(self,max_bound, steps,**kargs):
        if len(steps)>1: 
            print('warning: give two or more steps in cosine lr decay')
            print('warning: The cosine lr decay will only use the first one')
        self.steps = steps[0]
        self.max_bound = max_bound
    def __call__(self,iter):
        return 0.5*(np.cos((iter-self.steps)/(self.max_bound-self.steps)*np.pi)+1) if iter>self.steps else 1.0

class step_lrdecay(object):
    def __init__(self,decay_ratio,steps,**kargs):
        self.steps = np.array(steps).astype(int)
        self.decay_ratio = decay_ratio
    def __call__(self,iter):
        return self.decay_ratio**(iter>self.steps).sum()  

class General_WarmUP(object):
    def __init__(self, 
                type, 
                epoch_iter, 
                bound=0, 
                bound_unit='epoch', 
                step_type='epoch', 
                ratio=1.0,
                **kargs):
        assert bound_unit in ['epoch','iter']
        assert step_type in ['epoch','iter']
        self.step_type = step_type
        self.bound = bound
        self.epoch_iter = epoch_iter
        self.ratio = ratio
        if bound_unit=='epoch':
            self.bound *= epoch_iter
        self.warmup_func = globals()[type+'_warmup']
    def __call__(self,iter):
        if self.step_type=='iter':
            wup_ratio = self.warmup_func(iter,self.bound)
        elif self.step_type=='epoch': 
            wup_ratio = self.warmup_func(int(iter/self.epoch_iter),int(self.bound/self.epoch_iter))
        return self.ratio+(1-self.ratio)*wup_ratio

class General_LrDecay(object):
    def __init__(self, type, epoch_iter, max_epoch, steps, steps_unit='epoch', step_type='epoch', **kargs):
        assert step_type in ['epoch','iter']
        assert steps_unit in ['epoch','iter']
        self.step_type = step_type
        self.epoch_iter = epoch_iter
        max_bound = max_epoch*epoch_iter
        if steps_unit=='epoch':
            steps = [_*epoch_iter for _ in steps]
        if step_type=='epoch':
            max_bound = max_epoch
            steps = [_/epoch_iter for _ in steps]
        self.lrdecay_func = globals()[type+'_lrdecay'](max_bound=max_bound, steps=steps,**kargs)
    def __call__(self,iter):
        if self.step_type=='iter':
            return self.lrdecay_func(iter+1)
        elif self.step_type=='epoch': 
            return self.lrdecay_func(int(iter/self.epoch_iter)+1)

class Scheduler(object):
    def __init__(self,epoch_iter, max_epoch, warm_up=None, lr_decay=None, **kargs):
        if warm_up is not None:
            warm_up_func = General_WarmUP(epoch_iter=epoch_iter,**warm_up)
            warmup_bound = warm_up_func.bound
        else:
            warm_up_func = lambda iter: 1.0
            warmup_bound = 0
        if lr_decay is not None:    
            lr_decay_func = General_LrDecay(epoch_iter=epoch_iter, max_epoch=max_epoch, **lr_decay)
        else:
            lr_decay_func = lambda iter: 1.0
        self.lr_lambda = lambda iter: warm_up_func(iter) if iter<=warmup_bound else lr_decay_func(iter)
    def __call__(self,iter):
        return self.lr_lambda(iter)

if __name__=='__main__':
    ep_it = 28
    max_ep = 40
    linear_decay = dict(
                      type='step',
                      step_type='epoch',
                      decay_ratio=0.1,
                      steps=[20,30],
                      steps_unit='epoch')
    lr_sch = Scheduler(ep_it,max_ep,
                 warm_up = dict(
                      type='linear',
                      ratio=0.0,
                      step_type='iter',
                      bound=6, 
                      bound_unit='epoch',
                 ),
                 lr_decay=dict(
                      type='cos',  #cos, step
                      step_type='iter',
                      # decay_ratio=0.1,  # step decay parameters
                      steps=[20],
                      steps_unit='epoch') )
    for e in range(max_ep):
        for i in range(ep_it):
            print('Epoch %d [%d/%d]: %.8f'%(e,i,ep_it,lr_sch(e*ep_it+i)))