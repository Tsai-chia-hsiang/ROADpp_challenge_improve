import json
from pathlib import Path
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from distinguish.dataset import MC_ReID_Features_Dataset
from distinguish.model import MODELS, _Contrastive_Learning_Model
from distinguish.dataset import MC_ReID_Features_Dataset
from distinguish.loss import set_seed
from distinguish.log import get_logger, remove_old_tf_evenfile
from torch.utils.tensorboard import SummaryWriter

def layze_parse_arg():
    """
    My Lam-Par serrrrrrrr
    """
    parser = ArgumentParser()
    parser.add_argument("--operation", type=str,nargs='+', default=['train'])
    parser.add_argument("--model", type=str, default="vit")
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path("./")/"crop")
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt")/"cls")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--valid_epochs", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--debug_iter", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=891122)
    args = parser.parse_args()
    return args

def test_cls(args, ncls:int=10, mode:str="valid"):
   
    dev = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")

    assert args.pretrained is not None
    dset = MC_ReID_Features_Dataset.build_dataset(
        root=Path(args.root)/f"{mode}"
    )
    model:_Contrastive_Learning_Model = MODELS[args.model](
        ncls=ncls, ckpt= args.pretrained
    )
    model.to(device=dev)

    eva = model.inference_one_epoch(
        inference_loader= DataLoader(
            dataset=dset, batch_size=args.batch_size
        ),
        return_pred=False, metrcs_tolist=True, confusion_matrix="pd",
        dev=dev
    )
    with open(args.ckpt/f"valid_metrics.json", "w+") as f:
        json.dump(
            {k:v for k, v in eva.items() if k != "confusion matrix"}, 
            f, indent=4, ensure_ascii=False
        )
    
    eva['confusion matrix'].to_csv(args.ckpt/'confusion_matrix.csv', index=False)

def train_cls(args):
    
    set_seed(args.seed)
    Path(args.ckpt).mkdir(parents=True, exist_ok=True)
    logger = get_logger(name=__name__, file=args.ckpt/"training.log")
    remove_old_tf_evenfile(args.ckpt)
    logger.info(f"Settings : ")
    logger.info(f"{vars(args)}")
    logger.info(f"")

    dev = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    
    logger.info(f"Using : {torch.cuda.get_device_properties(dev)}")
    logger.info("") 
    logger.info(f"build training set from {Path(args.root)/'train'}")
    train_dataset = MC_ReID_Features_Dataset.build_dataset(root=Path(args.root)/"train")
    logger.info(f"training set class counts : ")
    logger.info(f"{train_dataset.cls_count.tolist()}")
    logger.info(f"")
    logger.info(f"build validation set from {Path(args.root)/'valid'}")
    valid_dataset = MC_ReID_Features_Dataset.build_dataset(root=Path(args.root)/"valid")
    logger.info(f"validation set class counts : ")
    logger.info(f"{valid_dataset.cls_count.tolist()}")
    logger.info(f"")
    model:_Contrastive_Learning_Model = MODELS[args.model](
        ncls=train_dataset.ncls,
        ckpt= args.pretrained
    ) 
    
    board = SummaryWriter(args.ckpt)
    model.train_model(
        train_set=train_dataset, valid_set=valid_dataset, dev=dev, 
        batch=args.batch_size, epochs=args.epochs, warm_up=args.warmup_epochs,
        optimizer=args.optimizer, lr=args.lr, contrastive_learning=True,
        ckpt = args.ckpt/"cls_vit", val_epochs = args.valid_epochs, 
        logger=logger, board=board, debug=args.debug_iter       
    )


if __name__ == "__main__":
    
    args = layze_parse_arg()
    
    if 'train' in args.operation:
        print("train")
        train_cls(args=args)
    
    if 'valid' in args.operation:
        print("valid")
        test_cls(args=args, mode='valid')
