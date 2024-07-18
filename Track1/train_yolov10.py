from ultralytics import YOLO
from pathlib import Path
import shutil
import os

def main(project, name, pretrained_ckpt=None):
    ckpt = pretrained_ckpt if pretrained_ckpt is not None else "yolov10l.pt"
    detector = YOLO(ckpt)
    detector.train(
        data="track1_detect_only.yaml",mode="detect",
        epochs = 50, imgsz = 1280, batch = 12,
        device=[2,3,4,5],
        project=project, name=name,
        single_cls=True, 
        patience = 50, 
        optimizer = "Adam"
    )

if __name__ == "__main__":

    project = "ckpt"
    name = "yolov10_detection_only_no_early_stop"
    
    model_dir = Path(project)/name
    if model_dir.is_dir():
        shutil.rmtree(model_dir)
    
    main(project=project, name=name)
   