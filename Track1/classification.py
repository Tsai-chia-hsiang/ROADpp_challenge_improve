import os
from pathlib import Path
from argparse import ArgumentParser
import torch
from distinguish.model import ViT_classifier
from distinguish.dataset import MC_ReID_Features_Dataset

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="vit")
    parser.add_argument("--root", type=Path, default=Path("../../")/"roadpp"/"crop")
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt")/"cls")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    train_dataset = MC_ReID_Features_Dataset.build_dataset(root=Path(args.root)/"train")
    valid_dataset = MC_ReID_Features_Dataset.build_dataset(root=Path(args.root)/"valid")
    model = ViT_classifier(ncls=train_dataset.ncls)
    

if __name__ == "__main__":
    main()


