import argparse
import os
import shutil
import glob
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import zipfile
import numpy as np
import cv2
import pickle
import gc
from PIL import Image
import torch
from ultralytics.engine.model import Model
from ultralytics import YOLO, RTDETR
from longtail_cls.distinguish.model import _Contrastive_Learning_Model
from longtail_cls.distinguish.model import MODELS as CLS_MODELS
from longtail_cls.distinguish.dataset import rgb_normalizor

def check_cuda():
    try:
        if torch.cuda.is_available():
            print("CUDA is available!")
            device = torch.device("cuda:0")
            print(f"Using device: {device}")
            properties = torch.cuda.get_device_properties(device)
            print(f"Device Name: {properties.name}")
            print(f"Total Memory: {properties.total_memory}")
        else:
            print("CUDA is not available")
    except Exception as e:
        print(f"An error occurred: {e}")

def crop(img:np.ndarray, x1:int, x2:int, y1:int, y2:int) -> torch.Tensor:
    c = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return rgb_normalizor(Image.fromarray(c))

def out_of_range(x:float, y:float, max_x:float|int, max_y:float|int)->tuple[float, float]:
    x = min(max(x, 0), max_x)
    y = min(max(y, 0), max_y)
    return x, y

def norm_box_into_absolute(bbox, img_w, img_h):
    return bbox * np.array([img_w, img_h, img_w, img_h])

def bbox_normalized(bbox, img_w, img_h):
    return bbox / np.array([img_w, img_h, img_w, img_h])

def zip_tube_file(file_path:Path):

    zip_path = file_path.parent/f"{file_path.stem}.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add the file to the zip archive
        zipf.write(file_path,file_path.name)

    
@torch.no_grad()
def make_tube(args, tracker, second_cls: Optional[_Contrastive_Learning_Model] = None, cls_map:Optional[dict]=None, semantic_map:Optional[dict]=None):
    """
    Make submit tube using track algorithm.
    
    Args:
        video_name (str): video name, ex: val_00001, train_00001...etc
        tracker (object): Yolov8's track result.
        video_shape (tuple): video's shape.
        t2_input_shape (tuple): track2 input shape.
        submit_shape (tuple): final submit shape.
    
    tube:
        tube['agent']['video_name'][idx]: {
            'label_id': class index, 
            'scores': bounding box scores, 
            'boxes': bounding box coordinates (absolute), 
            'score': tube score(we using np.mean(scores)), 
            'frames': frames across which the tube spans
        }
    """
    
    def tube_interpolation(tube):
        frames = tube['frames']
        scores = tube['scores']
        boxes = tube['boxes']
        
        interpolated_frames = np.arange(frames[0], frames[-1] + 1)  
        interpolated_scores = np.interp(interpolated_frames, frames, scores)  
        interpolated_boxes = np.empty((len(interpolated_frames), 4))  
        
        for i, axis in enumerate([0, 1, 2, 3]):
            interpolated_boxes[:, i] = np.interp(interpolated_frames, frames, boxes[:, axis])
        
        tube['boxes'] = interpolated_boxes
        tube['scores'] = interpolated_scores
        tube['frames'] = interpolated_frames
 
    def tube_change_axis(tube, orig_shape, submit_shape):
        ori_h, ori_w = orig_shape
        new_h, new_w = submit_shape
        
        tube['boxes'] = np.array(
            [
                norm_box_into_absolute(
                    bbox_normalized(box, ori_w, ori_h), new_w, new_h
                ) 
                for box in tube['boxes']
            ]
        )
    
    
    tracklet = {}
    frame_num = 0
    
    # Tracker.boxes.data(Tensor): x1, y1, x2, y2, track_id, conf, label_id
    for t in tracker:
        frame_num += 1
        if t.boxes.is_track:
            frame_img = t.orig_img
            for b in t.boxes.data:
                x1, y1, x2, y2, track_id, conf, label_id = b
                
                # Convert tensor values to Python scalars
                x1, y1, x2, y2, track_id, conf, label_id = (
                    x1.item(), y1.item(), x2.item(), y2.item(),
                    int(track_id.item()), 
                    conf.item(), 
                    int(label_id.item())
                )

                x1, y1 = out_of_range(x1, y1, t.orig_shape[1], t.orig_shape[0])
                x2, y2 = out_of_range(x2, y2, t.orig_shape[1], t.orig_shape[0])

       
                if track_id not in tracklet:
                    # agent
                    tracklet[track_id] = {
                        'label_id': label_id,
                        'scores': np.array([conf]),
                        'boxes': np.array([[x1, y1, x2, y2]]),
                        'score': 0.0,
                        'frames': np.array([frame_num]),
                    }
                    if args.two_stage:
                        tracklet[track_id]['crop']=crop(
                            frame_img, 
                            x1 = int(x1), x2=int(x2),
                            y1= int(y1), y2=int(y2)
                        )

                else:
                    # agent
                    tracklet[track_id]['scores'] = np.append(tracklet[track_id]['scores'], conf)
                    tracklet[track_id]['boxes'] = np.append(tracklet[track_id]['boxes'], [[x1, y1, x2, y2]], axis=0)
                    tracklet[track_id]['frames'] = np.append(tracklet[track_id]['frames'], frame_num)
                    
                    # using highest conf score to do second stage classificaiton
                    if args.two_stage:
                        if conf > max(tracklet[track_id]['scores']):
                            tracklet[track_id]['crop'] = crop(
                                frame_img, 
                                x1 = int(x1), x2=int(x2),
                                y1= int(y1), y2=int(y2)
                            )

    agent_list = []
    for tube_id, d in tracklet.items():
        tube_data = d.copy()
        tube_interpolation(tube_data)
        tube_change_axis(tube_data, args.video_shape, args.submit_shape) # change axis to submit_shape
        tube_data['score'] = np.mean(tube_data['scores'])
        
        if args.two_stage:
            
            if tube_data['label_id'] == cls_map['recls']['src']:
                re_label = second_cls.unit_inference(
                    x = tube_data['crop'],
                    dev = torch.device(f"cuda:{args.devices}")
                )
                re_label =  cls_map['recls']['target'][re_label]
                tube_data['label_id'] = re_label

            else:
                tube_data['label_id'] = cls_map['map'][tube_data['label_id']]
            
            del tube_data['crop']
            gc.collect()
        
        elif args.two_semantic:
            tube_data['label_id'] = semantic_map[tube_data['label_id']]
        
        agent_list.append(tube_data.copy())
    
    return agent_list



DETECTOR = {
    'yolo':YOLO,
    'rtdetr':RTDETR
}

TWO_STAGE_MAP = {
    'map':{
        0:0,
        1:1,
        2:2,
        3:3,
        5:7,
        6:9
    },
    'recls':{
        'src':4,
        'target':{
            0:4,
            1:5,
            2:6,
            3:8
        }
    }
}

TWO_SEMANTIC_MAP={
    'veh':{
        0:4,
        1:5,
        2:6,
        3:8
    },
    'other':{
        0:0,
        1:1,
        2:2,
        3:3,
        4:7,
        5:9
    }
}



def arg_parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, default="yolo")
    
    #data and result dir
    parser.add_argument('--video_path', type=str, default='./roadpp/test_videos', help='video path')
    parser.add_argument('--pkl_dir', type=Path, default=Path("./roadpp/submit"), help='submit file name(*.pkl)')
 
    # single branch detection
    parser.add_argument('--model_path', type=str, default='./submit/yolo10_e2/yolo.pt', help='yolo path')    
    # two stage classification with single branch detection
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument('--cls_model', type=str, default="resnext101")
    parser.add_argument('--cls_ckpt', type=Path, default="./longtail_cls/ckpt/cls_veh/resnext101_epoch61.pt")

    # two branch detection
    parser.add_argument("--two_semantic", action='store_true')
    parser.add_argument("--veh_model", type=Path)
    parser.add_argument("--other_model", type=Path)
    

    parser.add_argument('--devices', nargs='+', type=str, default='0', help='gpu number')
    parser.add_argument('--imgsz', type=tuple, default=(1280, 1280), help='yolo input size')
    parser.add_argument('--video_shape', type=tuple, default=(1280, 1920), help='original video resolution')
    parser.add_argument('--submit_shape', type=tuple, default=(600, 840), help='final submit shape')
    
    opt = parser.parse_args()
    assert opt.detector.lower() in DETECTOR
    assert not (opt.two_stage and opt.two_semantic), "Can only use two stage or two semantic"
    print(opt)
    check_cuda()
    _ = input("OK ?")
    return opt


if __name__ == "__main__":
    
    args = arg_parse()
    args.pkl_dir.mkdir(parents=True, exist_ok=True)

    tube = {'agent':{}}
    det:Model = None
    second_cls = None
    det_veh:Model = None
    det_other:Model =None
    
    print("Loaing ckpt ..")
    if not args.two_semantic:
        print(f"single branch {args.detector} from {args.model_path}")
        shutil.copy(args.model_path, args.pkl_dir/f'{args.detector}.pt')
        det = DETECTOR[args.detector](args.model_path)
        
        if args.two_stage:
            print(f"using two stage classification for top class {TWO_STAGE_MAP['recls']['src']} using {args.cls_model} : {args.cls_ckpt}")
            shutil.copy(args.cls_ckpt, args.pkl_dir/f'sec_cls_{args.cls_model}.pt')
            second_cls = CLS_MODELS[args.cls_model](
                ckpt = args.cls_ckpt,
                ncls = len(TWO_STAGE_MAP['recls']['target']),
            )
           
    else:
        print(f"two branch {args.detector}, veh :{args.veh_model}, other :{args.other_model}")
        shutil.copy(args.veh_model, args.pkl_dir/f"{args.detector}_veh.pt")
        shutil.copy(args.other_model, args.pkl_dir/f"{args.detector}_other.pt")
        det_veh = DETECTOR[args.detector](args.veh_model)
        det_other = DETECTOR[args.detector](args.other_model)
         
    print(args.video_path)
    all_vs = sorted(glob.glob(os.path.join(args.video_path, '*.mp4')))
    for idx, v in enumerate(tqdm(all_vs)):
        video_name = v.split('/')[-1].split('.')[0]
        print(v, video_name)
        if not args.two_semantic:
            tracker = det.track(     
                source=v, imgsz=args.imgsz,
                device=args.devices, stream=True,
                conf = 0.0, verbose=False
            )
            tube['agent'][video_name] = make_tube(
                args=args, tracker=tracker,
                second_cls=second_cls, cls_map=TWO_STAGE_MAP
            )
        else:
            veh_tracker = det_veh.track(     
                source=v, imgsz=args.imgsz,
                device=args.devices, stream=True,
                conf = 0.0, verbose=False
            )
            print(f"tracking veh ..")
            tube_veh = make_tube(
                args=args, tracker=veh_tracker, 
                semantic_map=TWO_SEMANTIC_MAP['veh']
            )
            other_tracker = det_other.track(
                source=v, imgsz=args.imgsz,
                device=args.devices, stream=True,
                conf = 0.0, verbose=False
            )
            print(f"tracking the others ..")
            tube_other = make_tube(
                args=args, tracker=other_tracker, 
                semantic_map=TWO_SEMANTIC_MAP['other']
            )

            tube['agent'][video_name] = tube_veh + tube_other
    
    tube_file_name = args.pkl_dir/f"{args.pkl_dir.parts[-1]}_tubes.pkl"
    print(f"writing {tube_file_name} ..")
    with open(tube_file_name, 'wb+') as f:
        pickle.dump(tube, f)
    print("zipping ..")
    zip_tube_file(file_path=tube_file_name)
    