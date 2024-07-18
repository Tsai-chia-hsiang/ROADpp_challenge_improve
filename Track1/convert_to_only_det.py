from pathlib import Path
from tqdm import tqdm
import shutil

def convert_a_task(src_root:Path = Path("/mnt/Nami/dataset/roadpp"), dst_root:Path=Path("../roadpp/detection_only/"), t:str="train"):
    root = src_root/t/"labels"
    dst = dst_root/t/"labels_cls"
    print(f"{root} -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    anns = sorted([_ for _ in Path(root).glob("*.txt")])
    for ai in tqdm(anns):
        #shutil.copy(ai, dst/f"{ai.stem}.txt")
        I = open(ai, "r")
        buf = [_.strip().split(" ") for _ in I.readlines()]
        I.close()
        q = {tuple(lst) for lst in buf}
        O = open(dst/f"{ai.stem}.txt", "w+")
        for bi in list(q):
            print(" ".join(list(bi)), file=O)
        O.close()

def move_image(src_root:Path = Path("/mnt/Nami/dataset/roadpp"), dst_root:Path=Path("../roadpp/detection_only/"), t:str="train"):
    root = src_root/t/"images"
    dst = dst_root/t/"images"
    print(f"{root} -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    imgs = sorted([_ for _ in Path(root).glob("*.jpg")])
    for i in tqdm(imgs):
        pure_name = f"{i.stem}.jpg"
        shutil.copy(i, dst/pure_name)
        


def main():
    for task in ["train", "valid"]:
        convert_a_task(t=task)
        #move_image(t=task)


if __name__ == "__main__":
    main()