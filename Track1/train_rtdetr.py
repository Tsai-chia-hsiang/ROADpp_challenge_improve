import os
#os["CUDA_VISIBLE_DEVICES"]="4,5"
import torch
from ultralytics import RTDETR
from pathlib import Path
import shutil

def resume(last_ckpt:str="./ckpt/rtdetr_roadpp_e2e/weights/last.pt"):
    model = RTDETR(last_ckpt)
    model.train(resume=True)
    

def main(project, name):
   
    model = RTDETR("rtdetr-l.pt")
    
    model.train(
        data="track1.yaml",mode="detect",
        epochs = 50, imgsz = 1280, batch = 4,
        device=[6, 7], deterministic=False,
        amp=False, project=project, name=name,
        patience = 10, optimizer="AdamW", 
        lr0=0.001
    )

if __name__ == "__main__":
    project = "ckpt"
    name = "rtdetr"
    model_dir = Path(project)/name
    if model_dir.is_dir():
        print(f"rm {model_dir}")
        shutil.rmtree(model_dir)
    main(project=project, name=name)
    