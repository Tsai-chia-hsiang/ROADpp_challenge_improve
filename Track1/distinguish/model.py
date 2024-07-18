import torch
from torchvision import models
from typing import Literal

def ResNeXt101_encoder() -> tuple[torch.nn.Sequential, int]:
    encoder = models.resnext101_32x8d(pretrained=True)
    out_f = encoder.fc.in_features
    return torch.nn.Sequential(*list(encoder.children())[:-1]), out_f
def ViT_encoder(with_head = False)->torch.nn.Module|tuple[torch.nn.Module, ]:
    model = models.vit_b_16(pretrained=True)
    if not with_head:
        return model.encoder
    return model.encoder, model.heads

EN_MAP ={
    'resnext101':ResNeXt101_encoder
}


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


class CLS_Model(torch.nn.Module):

    def __init__(self, ncls:) -> None:
        super().__init__(*args, **kwargs)

