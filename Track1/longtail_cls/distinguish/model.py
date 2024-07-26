"""
I have modified the code from VisionTransformer forward() founction
To let it return token if give the argument ```with_token = True``
"""
import os
import logging
from logging import Logger
from typing import Literal, Optional, Callable, Iterable
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
from torch.optim import Optimizer, Adam, AdamW, SGD
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.vision_transformer import VisionTransformer
from .scheduling import adjust_lr
from .loss import LogitAdjust, SCL
from .dataset import MC_ReID_Features_Dataset
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torch.utils.tensorboard import SummaryWriter


def ResNeXt101_encoder() -> tuple[torch.nn.Sequential, int]:
    encoder = models.resnext101_32x8d(pretrained=True)
    out_f = encoder.fc.in_features
    return torch.nn.Sequential(*list(encoder.children())[:-1]), out_f

def ViT_encoder(with_head = False)->torch.nn.Module|tuple[torch.nn.Module, ]:
    model = models.vit_b_16(pretrained=True)
    if not with_head:
        return model.encoder
    return model.encoder, model.heads

EN_MAP ={'resnext101':ResNeXt101_encoder}

def remove_module_prefix(state_dict):
    """
    Remove the 'module.' prefix from the state dictionary keys.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove the 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

class ReID_Model(torch.nn.Module):
    
    def __init__(self, ncls:int, fe:Literal["resnext101"] = "resnext101") -> None:
        super().__init__()
        self.encoder, self.feat_dim = EN_MAP[fe]()
        
        self.cls_head = torch.nn.Linear(self.feat_dim, ncls)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        x : (Nx3, 3, H ,W)
        Nx3 : [anchor, inter_pos, intra_pos]
        """
        ebd = self.encoder(x)


class ViT_Cls_Constractive_Model(torch.nn.Module):

    def __init__(self, ncls:int=10) -> None:
    
        super().__init__()
    
        self.ncls = ncls
        self.backbone:VisionTransformer = models.vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT
        )
        self.fdim = 128
        self.backbone.heads.head = torch.nn.Linear(self.backbone.heads.head.in_features, ncls)
        self.feature_net = torch.nn.Sequential(
            *[
                torch.nn.Linear(768, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 128)
            ]
        )

    def forward(self, x:torch.Tensor, with_token:bool=False):
        out = self.backbone(x, with_token=with_token)
        if with_token:
            f = self.feature_net(out[1][:, 0])
            return out[0], F.normalize(f, p=2, dim=1)
        return out
    
    @classmethod
    def build_model(cls, ncls:int, ckpt:Optional[os.PathLike]=None):
        
        M = cls(ncls=ncls)
        if ckpt is not None:
            print(f"load pretrained weights from {ckpt}")
            M.load_state_dict(remove_module_prefix(torch.load(ckpt, map_location="cpu")))
        
        return M
    
    def _get_optimizer(self, opt:Literal["adam","adamw", "sgd"]="adamw", lr:float=0.01, momentum:float=0.9)->Optimizer:

        match opt.lower():
            case "adam":
                return Adam(self.parameters(), lr=lr)
            case "adamw":
                return AdamW(self.parameters(), lr=lr)
            case "sgd":
                return SGD(self.parameters(), lr=lr, momentum=momentum)
            case _:
                raise KeyError(f"Not support {opt}")
    
    @torch.no_grad
    def get_prototype(
        self, dset:MC_ReID_Features_Dataset, 
        batch:int=50, dev:torch.device=torch.device("cpu"),
        logger:Optional[Logger]=None
    ) -> torch.Tensor:
        
        if logger is not None:
            logger.info("get prototypes for each class from dset")
        else:
            print("get prototypes for each class from dset")
        
        ptype = torch.zeros((self.ncls, self.fdim)).to(device=dev).detach()
        loader = DataLoader(dset, batch_size=batch, shuffle=False)
        cls_num = dset.cls_count.to(device=dev)
        
        xi:torch.FloatTensor = None
        yi:torch.LongTensor = None

        for xi, yi in tqdm(loader):
            _, f = self(xi.to(dev), with_token=True)
            for li in yi.unique(return_counts=False):
                ptype[li] += torch.sum(
                    (f[torch.where(yi == li)[0]]/cls_num[li]),
                    dim=0
                ).detach()

        return ptype

    def train_model(
        self, logger:Logger,
        train_set:MC_ReID_Features_Dataset, 
        valid_set:Optional[MC_ReID_Features_Dataset]=None, 
        dev:torch.device = torch.device("cpu"), batch:int=50,
        epochs:int=20, warm_up:int=2, val_epochs:int=2,
        optimizer:Literal["adam","adamw", "sgd"]="adamw", lr:int=0.01,
        contrastive_learning:bool=False,
        ckpt:Path = Path("vit_cls"),
        board:Optional[SummaryWriter]=None,
        debug:int=-1     
    ) -> Optimizer:
                
        self.to(device=dev)
        cls_loss = LogitAdjust(cls_num_list=self.ncls, device=dev, weight=torch.log(train_set.cls_w))
        scl_loss = SCL(self.ncls, device=dev, fdim=self.fdim)
        if warm_up == -1 and debug <= 0:
            logger.info("using pretrained weights to build prototype")
            if not (ckpt/f"init_prototype.pt").is_file():
                scl_loss.prototype = self.get_prototype(
                    dset=train_set, logger=logger, 
                    dev=dev, batch=batch
                )
                logger.info(f"save to {ckpt/f'init_prototype.pt'}")
                torch.save(scl_loss.prototype.cpu(), ckpt/f"init_prototype.pt")
            else:
                logger.info(f"load from {ckpt/f'init_prototype.pt'}")
                scl_loss.prototype = torch.load(ckpt/f"init_prototype.pt")
        
        logger.info(f"Training VIT classification model {epochs} epochs with CE ,{'contrastive loss' if contrastive_learning else ''} ")
        logger.info(f"optimizer : {optimizer} with initial lr {lr}")
        
        optim:Optimizer = self._get_optimizer(opt=optimizer, lr=lr)
        
        train_loader = DataLoader(
            dataset=train_set, 
            batch_size=batch, 
            shuffle=True, 
            sampler=train_set.bias_sampler,
            pin_memory=True
        )
        valid_loader = DataLoader(
            dataset=valid_set, 
            batch_size=batch, 
            shuffle=False,
            pin_memory=True
        ) if valid_set is not None else None

        if valid_set is None:
            logger.info("===> No validation set, using training loss to judeg <===")
        
        best_f1 = 0
        best_loss = np.inf
        
        for e in range(epochs):
        
            if debug > 0:
                logger.info(f"debugging, run just {debug} batch(s)")
            
            if e < warm_up:
                logger.info(f"training epoch {e}, warm-up : only classification loss")
            else :
                logger.info(f"training epoch {e}, with contrastive learning")

            loss_log = self._train_one_epoch(
                train_loader=train_loader,optim=optim, dev=dev, 
                cls_criteria=cls_loss, logger=logger,
                contrastive_learning=contrastive_learning if e >= warm_up else False,
                contrastive_criteria=scl_loss,
                board=board,
                debug_iter=debug
            )
            logger.info(f"training loss : ")
            logger.info(f"{loss_log}")
            if valid_loader is not None and ( (e+1)%val_epochs == 0 or (e+1) == warm_up):
                
                logger.info(f"validation epoch {e}")
                valid_log = self.inference_one_epoch(
                    inference_loader=valid_loader,
                    dev=dev, return_pred=False,
                    debug_iter=debug
                )

                logger.info(f"validation F1 : {valid_log['f1']}; macro : {valid_log['macro f1']}")
                
                if board is not None:
                    board.add_scalar(f"valid_macro_f1", valid_log['macro f1'], e)
                
                if valid_log['macro f1'] >= best_f1:
                    save_to = ckpt.parent/f'{ckpt.stem}_epoch{e}.pt'
                    logger.info(f"current best valid f1:{best_f1}; new best valid f1:{valid_log['macro f1']}")
                    logger.info(f"save weights to {save_to}")
                    torch.save(self.state_dict(), save_to)
                    best_f1 = valid_log['macro f1']
                
            else:
                if (e+1)%val_epochs == 0 and loss_log['total'] <= best_loss:
                    logger.info(f"current best loss:{best_loss}; new best loss:{loss_log['total']}")
                    logger.info(f"save weights to {save_to}")
                    
                    torch.save(self.state_dict(), ckpt.parent/f"{ckpt.stem}_no_valid_epoch{e}.pt")
                    best_loss = loss_log['total']
        
        return optim
    
    def _train_one_epoch(
        self, train_loader:DataLoader,  logger:Logger,
        optim:Optimizer, dev:torch.device, cls_criteria:LogitAdjust, 
        contrastive_learning:bool=False, contrastive_criteria:Optional[SCL]=None, 
        w:Iterable[float]=(2, 0.6), pbar:bool=True, board:Optional[SummaryWriter]=None, 
        log_batch_num:int=10, epoch:int=1, debug_iter:int=-1,
    ) -> float|dict[str, float]:
        
        img:torch.FloatTensor = None
        xi:torch.FloatTensor = None
        yi:torch.FloatTensor = None
        fi:torch.FloatTensor = None
        li:torch.LongTensor = None
        n_sample = 0
        log_freq = len(train_loader) // log_batch_num
        
        self.train()
        critera_cls, critera_cl, critera_total = 0, 0, 0
        current_cls, current_cl, current_total = 0, 0, 0

        bar = tqdm(train_loader) if pbar else train_loader
        for idx, (img, li) in enumerate(bar):
            optim.zero_grad()
            xi = img.to(device=dev)
            li = li.to(dev)
            yi, fi = self(xi, with_token=True)
            cls_l = cls_criteria(yi, li)

            feature_loss:torch.Tensor = 0.0
            if contrastive_learning:
                cls_l = cls_l*w[0]
                feature_loss = contrastive_criteria(fi, li)*w[1]
            
            total_loss:torch.Tensor = cls_l + feature_loss
            total_loss.backward()
            optim.step()
            
            n_sample += xi.size(0)
            critera_cls += cls_l.item()*xi.size(0)
            if contrastive_learning:
                critera_cl += feature_loss.item()*xi.size(0) 
                critera_total += total_loss.item()*xi.size(0)
                contrastive_criteria.update_prototype(pnew=fi.detach(),update_cls=li.detach())
        
            if idx % log_freq == 0:
                current_cls = critera_cls/n_sample
                loss_info = f"cls_loss: {current_cls:.4f} "
                if contrastive_learning:
                    current_cl = critera_cl/n_sample
                    current_total = critera_total/n_sample
                   
                    loss_info += f"cl_loss: {current_cl:.4f} total_loss: {current_total:.4f}"
                
                logger.info(loss_info, extra={'file_only':True})
            
                if board is not None:
                    board.add_scalar("cls_loss",current_cls, epoch * len(train_loader) + idx)
                    if contrastive_learning:
                        
                        board.add_scalar("constrastive_loss", current_cl, epoch * len(train_loader) + idx)
                        board.add_scalar("total_loss",current_total, epoch * len(train_loader) + idx)

            if pbar:
                bar.set_postfix(
                    ordered_dict={
                        'cls_loss': f"{cls_l.item():.4f}",
                        'cl':f"{feature_loss.item():.4f}",
                        'total_loss':f"{total_loss.item():.4f}"
                    } if contrastive_learning else {
                        'cls_loss' : f"{cls_l.item():.4f}"
                    }
                )
            if debug_iter > 0:
                if idx == debug_iter:
                    break

        return {
            'cls':critera_cls/n_sample,
            'total':critera_cls/n_sample
        } if not contrastive_learning else \
        {
            'cls':critera_cls/n_sample,
            'contrastive':critera_cl/n_sample,
            'total':critera_total/n_sample
        }
    
    @torch.no_grad()
    def inference_one_epoch(
        self, inference_loader:DataLoader, dev:torch.device, 
        pbar:bool=True, return_pred:bool=True, 
        cls_loss:Optional[LogitAdjust]=None,
        debug_iter:int=-1
    ) -> np.ndarray|dict|tuple[np.ndarray, dict]:
        
        xi:torch.FloatTensor = None
        gth:list[int] = []
        pred:list[int] = []
        
        bar = tqdm(inference_loader) if pbar else inference_loader
        inference_flage=False
        self.eval()
        for idx, i in enumerate(bar):
            if len(i) == 2:
                # validation 
                xi = i[0].to(dev)
                gth += i[1].tolist()
            elif len(i) == 1:
                # inference
                inference_flage = True
                xi = i.to(dev)

            pred += torch.argmax(self(xi), dim=1).cpu().tolist()
            
            if debug_iter > 0:
                if idx == debug_iter:
                    break

        pred= np.array(pred)
        if inference_flage:
            # no gth, just inference
            return pred
        
        gth = np.array(gth)
        
        f1 = f1_score(gth, pred, average=None)
        
        metrics = {
            'precision':precision_score(gth, pred, average=None),
            'recall':recall_score(gth, pred, average=None),
            'f1':f1,
            'macro f1':np.mean(f1)
        }

        if return_pred:
            return pred, metrics
        
        return metrics

    @torch.no_grad()
    def unit_inference(self,x:torch.Tensor,dev:torch.device)->int:
        self.eval()
        self.to(dev)
        y = torch.argmax(self(x.to(dev)), dim=1).cpu()
        return y.item()
    