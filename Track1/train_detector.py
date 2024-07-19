import os
import sys
import time
from typing import Literal
import os.path as osp
import torch
from ultralytics.engine.model import Model
from ultralytics import RTDETR, YOLO
from pathlib import Path
import shutil
from argparse import ArgumentParser

MODEL_MAP = {
    'yolo':YOLO,
    'rtdetr':RTDETR
}

def get_general_pretrained_ulra_model(arch:str):
    backbone = None
    if 'yolo' in arch:
        backbone = 'yolo'
    elif 'rtdetr' in arch:
        backbone = 'rtdetr'
    else:
        raise KeyError("Not support this arch")
    
    return MODEL_MAP[backbone](arch)

def parse_cmd_args()->tuple[str, dict, dict]:
    
    def confirm(modelarch:str, args:dict, train_args:dict) -> bool:
        indent = 50
        print()
        print("pathes settings:")
        print("="*indent)
        print(f"dataset     : {args['data_config']}")
        print(f"saving root : {args['project']}/{args['name']}/")
        if args['resume']:
            last_ckpt = Path(args['project'])/args['name']/"weights"/"last.pt"
            assert os.path.exists(last_ckpt)
            print(f"resume from {last_ckpt} -- {os.path.exists(last_ckpt)}")
        
        print("="*indent)
        print()
        print(f"training hyperparameters for {modelarch}")
        print("="*indent)
        for idx, (k, v) in enumerate(train_args.items()):
            kinfo = f"{k}"
            vinfo = f"{v}"
            info = kinfo + " "*(15-len(kinfo)) + ": " + vinfo 
            print(info, end=" "*(25 - len(info))+", " if idx %2 == 0 else "\n")
   
        if idx % 2 == 0:
            print()
        print("="*indent)
        print()
        check = "WAIT"
        while check not in ["y", "n", ""]:
            check = input("Sure? [y/Enter (Continue); n (Abort)]: ")
            if check != "":
                check = check.lower()
        return True if check in ["y", ""] else False
        
    parser = ArgumentParser()
    
    #detector backbone
    parser.add_argument("--detector", type=str, default="yolov10l.pt")
    parser.add_argument("--device",type=int,nargs='+', default=[0])

    #path 
    parser.add_argument("--project", type=Path, default="./ckpt")
    parser.add_argument("--data_config", type=Path, default="Track1/configs/track1.yaml")
    parser.add_argument("--name", type=str, default="")

    # hyperparameterss
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr0", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="")
    parser.add_argument("--no_deterministic", action='store_false')
    parser.add_argument("--imgsz",type=int, default=1280)
    parser.add_argument("--amp", action='store_true')
    parser.add_argument("--resume", action='store_true')

    args = vars(parser.parse_args())
    
    model_arch = args["detector"].lower()
    assert 'rtdetr' in model_arch or 'yolo' in model_arch
    
    path_args={k:v for k,v in args.items() if k in ["data_config", "project"]}
    if args["name"] == "":
        path_args['name'] = Path(model_arch).stem + "_" + Path(args["data_config"]).stem 
    else:
        path_args['name'] = args['name']

    path_args['resume'] = args['resume']

    train_args = {
        "epochs":50, "patience":-1, "batch":5, "device":0, "lr0":1e-2,
        "single_cls":False, "optimizer":"Adam", "imgsz":1280,
        "deterministic":True, 'amp':False
    }
    if 'detect_only' in Path(args['data_config']).stem:
        train_args['single_cls'] = True
    
    for k in train_args:
        
        if k in args:
            
            train_args[k] = args[k]
            
            if k == "device" and len(args[k]) == 1:
                train_args[k] = args[k][0]
            
            if k == "optimizer":
                if args[k] == "":
                    if "yolo" in model_arch:
                        train_args[k] = "Adam"
                    elif "rtdetr" in model_arch:
                        train_args[k] = "AdamW"
            
            if k == "patience":
                if args[k] == -1:
                    #do not early stop
                    train_args[k] = args["epochs"]

        elif k == "deterministic":
            train_args[k] = args["no_deterministic"]
            if "rtdetr" in model_arch:
                train_args[k] = False

    
    check = confirm(
        modelarch=model_arch, args=path_args, 
        train_args=train_args
    )
    
    if not check:
        print("Abort.")
        sys.exit(0)
    
    return model_arch, path_args, train_args
    
def resume(last_ckpt:str, arch:Literal["yolo", 'rtdetr']):
    print(f"{arch} <- {last_ckpt}")
    ultra_model:Model = MODEL_MAP[arch](last_ckpt)
    ultra_model.train(resume=True)

def train_ultra_model(ultra_model:Model, data_cfg:os.PathLike, name:str, project:os.PathLike="ckpt", **train_args):
   
    model_dir = Path(project)/name
    if model_dir.is_dir():
        print(f"rm {model_dir}")
        shutil.rmtree(model_dir)
    
    
    time.sleep(2)

    ultra_model.train(
        data=data_cfg, mode="detect",project=project, name=name,
        **train_args
    )
 
def val_ultra_model(ultra_model:Model, data_cfg:os.PathLike, name:os.PathLike, imgsz:int=1280, batch:int=5, device:int|list[int]=0):
    results = ultra_model.val(
        data = data_cfg,
        imgsz = imgsz,
        device = device,
        batch = batch,
        name = name
    )
    print(results)


"""
model = RTDETR("rtdetr-l.pt")
    
epochs = 50, imgsz = 1280, batch = 5,
        deterministic=False,
        single_cls=single_cls,
        amp=False, 
        patience = 30, optimizer="AdamW", 
        lr0=0.007

"""

def main():
    model_arch, args, train_args = parse_cmd_args()
    ultra_model = None
    if args['resume']:
        ckpt = Path(args['project'])/args['name']/"weights"/"last.pt"
        a = ""
        if 'yolo' in model_arch:
            a = 'yolo'
        elif 'rtdetr' in model_arch:
            a = 'rtdetr'
        
        resume(last_ckpt=ckpt, arch=a)    
        return 
    
    ultra_model = get_general_pretrained_ulra_model(arch=model_arch)
    train_ultra_model(
        ultra_model=ultra_model, 
        data_cfg=args["data_config"], 
        name=args['name'], project=args['project'],
        **train_args
    )
    

if __name__ == "__main__":
    main()