import os
from pathlib import Path
from argparse import ArgumentParser
import torch
from distinguish.dataset import MC_ReID_Features_Dataset
from distinguish.model import ViT_Cls_Constractive_Model
from distinguish.dataset import MC_ReID_Features_Dataset
from distinguish.loss import set_seed
from distinguish.loss import SCL
from distinguish.log import get_logger, remove_old_tf_evenfile
from torch.utils.tensorboard import SummaryWriter

def layze_parse_arg():
    """
    My Lam-Par serrrrrrrr
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="vit")
    parser.add_argument("--root", type=Path, default=Path("./")/"crop")
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt")/"cls")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--debug_iter", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=891122)
    args = parser.parse_args()
    return args

def main():
    
    args = layze_parse_arg()
    set_seed(891122)
    Path(args.ckpt).mkdir(parents=True, exist_ok=True)
    logger = get_logger(name=__name__, file=args.ckpt/"training.log")
    logger.info(f"Settings : ")
    logger.info(f"{vars(args)}")
    logger.info(f"")

    dev = torch.device(
        f"cuda:{max(args.device, torch.cuda.device_count()-1)}"
        if args.device >= 0 else "cpu"
    )
    
    #train loader
    logger.info(f"build training set from {Path(args.root)/'train'}")
    train_dataset = MC_ReID_Features_Dataset.build_dataset(root=Path(args.root)/"train")
    logger.info(f"training set class counts : ")
    logger.info(f"{train_dataset.cls_count.tolist()}")
    logger.info(f"")

    #validatino loader
    logger.info(f"build validation set from {Path(args.root)/'valid'}")
    valid_dataset = MC_ReID_Features_Dataset.build_dataset(root=Path(args.root)/"valid")
    logger.info(f"validation set class counts : ")
    logger.info(f"{valid_dataset.cls_count.tolist()}")
    logger.info(f"")
    
    #print device info
    logger.info("Device properties:")
    logger.info(f"{torch.cuda.get_device_properties(dev)}")
    logger.info("") 
    
    model = ViT_Cls_Constractive_Model(ncls=train_dataset.ncls)
    scl = SCL(model.ncls, device=dev)

    
    logger.info("warm-up, just classification")
    remove_old_tf_evenfile(args.ckpt)
    board = SummaryWriter(args.ckpt)
    current_opt = model.train_model(
        train_set=train_dataset, valid_set=valid_dataset, dev=dev, 
        batch=args.batch_size, epochs=args.warmup_epochs, 
        optimizer=args.optimizer, lr=args.lr, 
        scl_loss=scl, contrastive_learning=False,
        ckpt = args.ckpt/"cls_vit",
        val_epochs = 1, 
        logger=logger,board=board, 
        debug=args.debug_iter       
    )
    logger.info("with contrastive learning ..")
    model.train_model(
        train_set=train_dataset, valid_set=valid_dataset, dev=dev, 
        batch=args.batch_size, epochs=args.epochs - args.warmup_epochs, 
        shared_optimizer=current_opt,
        scl_loss=scl, contrastive_learning=True,
        ckpt = args.ckpt/"cls_vit",
        logger=logger, board=board,
        debug=args.debug_iter
    )


if __name__ == "__main__":
    main()


