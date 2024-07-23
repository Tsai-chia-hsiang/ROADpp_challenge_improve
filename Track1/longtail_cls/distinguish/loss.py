from typing import Callable
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def numerical_stable(logits:torch.Tensor) -> torch.Tensor:
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    return logits - logits_max.detach()

class LogitAdjust(Callable):

    def __init__(self, cls_num_list:torch.Tensor, tau=1, weight=None, device:torch.device=torch.device("cpu")):
        super().__init__()
        cls_num_list = torch.tensor(cls_num_list, device=device)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight.to(device)

    def __call__(self, x, target)->torch.Tensor:
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)
    
class SCL(Callable):

    def __init__(self, ncls:int, temperature=0.1, device=torch.device("cpu"), ema_alpha:float=0.99, fdim:int=768):
        
        super().__init__()
        self.on_device = device
        self.temperature = temperature
        self.ncls = ncls
        self.cls_indices = torch.arange(self.ncls, device=device).to(dtype=torch.int64)
        self._prototype:torch.Tensor = torch.zeros(ncls, fdim).detach().to(device=device)
        self.alpha = ema_alpha
    
    @property
    def prototype(self)->torch.Tensor:
        return self._prototype.clone()
    
    def update_prototype(self, pnew:torch.Tensor, update_cls:torch.Tensor)->None:
        """
        EMA updating with alpha = self.alpha
        """
        print("updating ..")
        for ci in update_cls.unique(return_counts=False):
            class_mean = torch.mean(pnew[torch.where(update_cls == ci)[0]], dim=0)
            self._prototype[ci] = \
                self.alpha*self._prototype[ci] + (1-self.alpha)*class_mean

    def __call__(self, features:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        
        if self.prototype is None:
            return 0
        bs = features.size(0)
        cls_index = torch.concat([targets, self.cls_indices]).detach()
        cls_count = torch.histc(
            cls_index.to(dtype=torch.float32), 
            bins=self.ncls, min=0, max=self.ncls-1
        ).view(1, -1).detach()
        
        class_one_hot_axis = F.one_hot(cls_index, self.ncls).to(
            dtype=torch.float32, device=self.on_device
        ).detach()
        
        #size :  N x (N + CLS) with self mask 
        cosine_map = (features@torch.vstack([features, self._prototype]).T/self.temperature).fill_diagonal_(0)
        cosine_map = numerical_stable(cosine_map).fill_diagonal_(0)
        
        # since there are prototypes for each class, cls_count can't be 0
        # at any class index (at least 1 : prototype for that class)
        cls_avg = torch.exp(cosine_map)@class_one_hot_axis/cls_count
        cls_avg = torch.sum(cls_avg, dim=1, keepdim=True)

        # postive samples filtering and self mask
        L = (cosine_map - torch.log(cls_avg))*(class_one_hot_axis[:, targets].T).fill_diagonal_(0)
        L = torch.sum(L,dim=1, keepdim=True)
        cl = -(1/(cls_count[0, targets] -1)).view(1, -1)@L
        return cl/bs

def set_seed(seed=42, loader=None):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True          

if __name__ == "__main__":

    set_seed(891122)
    scl = SCL(cls_prototype=torch.eye(3, dtype=torch.float32))
    
    test_sample = torch.randint(0,10, (5,3)).to(dtype=torch.float32)
    test_sample = F.normalize(test_sample, dim=1)
    target = torch.cat([torch.zeros(2), torch.ones(1), torch.ones(2)*2])
    target = target[ torch.randperm(target.size(0))].to(dtype=torch.int64)
    print(target)

    l = scl(features=test_sample, targets=target)

    x = 0