from pathlib import Path
from typing import Literal, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

rgb_normalizor = transforms.Compose(
    [transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )]
)

def n_count(l:torch.Tensor)->torch.Tensor:
    _, c = l.unique(return_counts=True)
    return c[torch.argsort(_)]

class MC_ReID_Features_Dataset(Dataset):
    
    def __init__(self, table:dict[int, dict[str, list[Path]]], cls_map:dict[str, int], mode:str="cls", bias_sampling:Optional[Literal["over","weighted_random"]]=None) -> None:
        
        super().__init__()
        
        self.label_to_clsid = cls_map.copy()
        self.clsid_to_label = {v:k for k,v in cls_map.items()}
        self.ncls = len(cls_map)
        
        self.T = rgb_normalizor
        
        self.patch_pathes, self.label, self.tid = self._flatten_table(data_table=table)
        
        self.total_samples = len(self.patch_pathes)
        self.indices = np.arange(self.total_samples)
        self.cls_count = n_count(torch.from_numpy(self.label).to(dtype=torch.float32))
        self.cls_w = self.total_samples/self.cls_count
        
        self.getitem_name = ["anchor", "inter_pos", "inter_neg"]#, "intra_neg"]
        self.getitem_sample_index = [None]*len(self.getitem_name)
        self.bias_sampling = bias_sampling
        self.bias_sampler = None
        
        if self.bias_sampling is not None:
            match self.bias_sampling:
                case "weighted_random":
                    w = (1/self.cls_count)[self.label]
                    self.bias_sampler = WeightedRandomSampler(
                        weights=w, num_samples=len(w),
                        replacement=True
                    )
                case "over":
                    raise NotImplementedError("TODO")
                case _:
                    raise KeyError(f"Not support {self.bias_sampling}")

        self.mode:str=mode

    @classmethod
    def build_dataset(cls, root:Path, mode:Literal["cls", "reid"]="cls", bias_sampling:Optional[Literal["over","weighted_random"]]=None):
        all_classes = sorted([_ for _ in root.iterdir() if _.is_dir()])
        cls_with_idx = list(zip(all_classes, range(len(all_classes))))
        table = {
            idx:[[i for i in tid.glob("*.jpg")] 
                 for tid in pi.iterdir()]
            for pi, idx in cls_with_idx
        }
        return cls(
            table=table, 
            cls_map = {k.name:v for k,v in cls_with_idx},
            mode=mode,
            bias_sampling=bias_sampling
        )  

    def _flatten_table(self, data_table:dict[int, list[list[str]]]) -> tuple[list[str], np.ndarray, np.ndarray]: 
        """
        Returns
        -------
        - patch path list : list of str
        - label : np ndarray with int64
        - tid : np ndarray with int64
        """

        pth = []
        l = []
        t = []
        for cls_id, p in data_table.items():
            for tid, pi in enumerate(p):
                pth += pi
                t += [tid]*len(pi)
                l += [cls_id]*len(pi)

        return pth, \
            np.asarray(l, dtype=np.int64), \
            np.asarray(t, dtype=np.int64)
    
    def __len__(self)->int:
        return self.total_samples
    
    def imread(self, p:str|Path|list[str|Path]=None)->torch.Tensor|None|list[torch.Tensor|None]:
        
        if p is None:
            return None
        
        elif isinstance(p, str) or isinstance(p, Path):
            return self.T(Image.open(p).convert("RGB"))
        
        elif isinstance(p, list):
            return [self.imread(_) for _ in p]
    
    def _reid_pair(self, index) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns
        ---
        >>> {sample_type : (img_path, patch tensor, cls_label tensor)}
        - sample_type includes :
            - a anchor
            - a inter-class postive sample (reid pos)
            - a inter-class negative sample (reid neg, cls pos) 
            - a intra-class negative sample (cls neg)
        """
        
        anchor_label = self.label[index]
        anchor_tid = self.tid[index]

        cls_mask = self.label == anchor_label
        tid_mask = self.tid == anchor_tid
        unique_mask = self.indices != index

        inter_neg_candidates = np.where(cls_mask & (~tid_mask) )[0]
        inter_pos_candidates = np.where(cls_mask & tid_mask & unique_mask)[0]

        #intra_neg_candidates = np.where(~cls_mask)[0]

        self.getitem_sample_index[0] = index
        self.getitem_sample_index[1] = np.random.choice(inter_pos_candidates) if len(inter_pos_candidates) else None
        self.getitem_sample_index[2] = np.random.choice(inter_neg_candidates)
        #self.getitem_sample_index[3] = np.random.choice(intra_neg_candidates)

        return {
            k:(
                self.patch_pathes[i] if i is not None else None,
                self.imread(self.patch_pathes[i]) if i is not None else None, 
                torch.tensor(self.label[i]) if i is not None else None
            ) 
            for k, i in zip(self.getitem_name, self.getitem_sample_index)  
        }
    
    def __getitem__(self, index):
        if self.mode == "cls":
            img = self.imread(self.patch_pathes[index])
            li = torch.tensor(self.label[index])
            #print(self.patch_pathes[index], img.size(), li.size())
            return img, li
        elif self.mode == "reid":
            return self._reid_pair(index=index)


if __name__ == "__main__":
    a_dataset = MC_ReID_Features_Dataset.build_dataset(root=Path("..")/"crop"/"train")
    test_iter = 5
    loader = DataLoader(dataset=a_dataset, batch_size=5)
    a_dataset.mode = "cls"
    for i, t in enumerate(loader):
        print(len(t))
        if len(t) == 2:
            #img & label
            print(t[0].size(), t[1])
        else:
            print(t)
        print(f"="*20)
        
        if i == 3:
            a_dataset.mode = "reid"

        elif i+1 == test_iter:
            break
    

        
        

        