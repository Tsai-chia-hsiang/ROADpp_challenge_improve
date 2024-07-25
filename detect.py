import os
import shutil

import sys
sys.path.append('./Track2')
import argparse
from pathlib import Path
import cv2
import glob
import torch
import pickle
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO, RTDETR
from tqdm import tqdm
from Track2.dataset import Tracklet_Dataset
from utils.tube_processing import zip_tube_file
from utils.linear_interpolation import tube_interpolation
from utils.tube_processing import tube_change_axis, action_tube_padding, combine_label, stack_imgs_padding


def out_of_range(x, y, max_x, max_y):
    x = min(max(x, 0), max_x)
    y = min(max(y, 0), max_y)
    return x, y

@torch.no_grad()
def make_tube(args):
    """
    Make submit tube using track algorithm.
    
    Args:
        tube (dict): Final submit data(See Submission_format.py)
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

        tube['triplet']['video_name'][idx]: {
            'label_id': class index, 
            'scores': bounding box scores, 
            'boxes': bounding box coordinates (absolute), 
            'score': tube score(we using np.mean(scores)), 
            'frames': frames across which the tube spans
            'stack_imgs': concated global & local img by frames
        }
    """
    
    tracklet = {}
    stack_imgs = {} 
    frame_num = 0
    
    # Tracker.boxes.data(Tensor): x1, y1, x2, y2, track_id, conf, label_id
    for t in args.tracker:
        frame_num += 1
        if t.boxes.is_track:
            frame_img = t.orig_img
            global_img = cv2.resize(
                cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), 
                args.t2_input_shape
            )

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

                if args.mode == 'Track2':
                    local_img = frame_img[int(y1) : int(y2), int(x1) : int(x2)]
                    local_img = cv2.resize(cv2.cvtColor(local_img, cv2.COLOR_BGR2RGB), args.t2_input_shape)
                    stack_img = np.concatenate((global_img, local_img), axis=-1)
                
                if track_id not in tracklet:
                    # agent
                    tracklet[track_id] = {
                        'label_id': label_id,
                        'scores': np.array([conf]),
                        'boxes': np.array([[x1, y1, x2, y2]]),
                        'score': 0.0,
                        'frames': np.array([frame_num])
                    }

                    # event
                    if args.mode == 'Track2':
                        stack_imgs[track_id] = [stack_img]
                else:
                    # agent
                    tracklet[track_id]['scores'] = np.append(tracklet[track_id]['scores'], conf)
                    tracklet[track_id]['boxes'] = np.append(tracklet[track_id]['boxes'], [[x1, y1, x2, y2]], axis=0)
                    tracklet[track_id]['frames'] = np.append(tracklet[track_id]['frames'], frame_num)

                    # event
                    if args.mode == 'Track2':
                        stack_imgs[track_id].append(stack_img)

    agent_list = []
    event_list = []
    
    for tube_id, tube_data in tracklet.items():
        # agent
        if args.mode == 'Track1': # if do interpolation in T2, len(tube_data['frames']) != len(stack_imgs[tube_id])
            tube_data = tube_interpolation(tube_data)
            
        tube_data = tube_change_axis(tube_data, args.video_shape, args.submit_shape) # change axis to submit_shape
        tube_data['score'] = np.mean(tube_data['scores'])
        agent_list.append(tube_data.copy())

        # event
        if args.mode == 'Track2':
            tube_data['stack_imgs'] = stack_imgs[tube_id]
            event_list.append(tube_data)
    
    if args.two_branch:
        return agent_list
    else:
        args.tube['agent'][args.video_name] = agent_list

    if args.mode == 'Track2':
        args.tube['triplet'][args.video_name] = event_list

    return 0


def make_t2_tube(tube, action_cls, loc_cls):
    t2_tubes = {}
    frames_len = len(tube['frames'])
    action_cls = action_tube_padding(
        action_cls,
        prev_frames=2,
        last_frames=1,
        frames_len=frames_len
    )
    
    combined_cls = []
    for frame_num in range(frames_len):
        combined_cls.append(combine_label(agent_id=tube['label_id'], action_id=action_cls[frame_num], loc_id=loc_cls[frame_num]))
        
    for frame_num in range(frames_len):
        cls = combined_cls[frame_num]
        if cls != -1:
            if cls not in t2_tubes:
                t2_tubes[cls] = {
                    'label_id': cls,
                    'scores': np.array([tube['scores'][frame_num]]),
                    'boxes': np.array([tube['boxes'][frame_num]]),
                    'score': tube['score'],
                    'frames': np.array([tube['frames'][frame_num]])
                }
            else:
                t2_tubes[cls]['scores'] = np.append(t2_tubes[cls]['scores'], tube['scores'][frame_num])
                t2_tubes[cls]['boxes'] = np.append(t2_tubes[cls]['boxes'], [tube['boxes'][frame_num]], axis=0)
                t2_tubes[cls]['frames'] = np.append(t2_tubes[cls]['frames'], tube['frames'][frame_num])

    t2_tubes_list = []
    for label_id, tube_data in t2_tubes.items():
        t2_tubes_list.append(tube_data)
        
    return t2_tubes_list


def track2(args):
    # ToDo: T2 interpolation bug
    event_tubes_list = []

    with torch.no_grad():
        with tqdm(args.tube['triplet'][args.video_name], desc="Processing tubes") as pbar:
            for t in pbar:
                # Create a dataset using Sliding Windows.
                action_dataset = Tracklet_Dataset(
                    mode='action',
                    tracklet=stack_imgs_padding(t['stack_imgs']), # padding when frames_num < 4
                    args=args
                )

                loc_dataset = Tracklet_Dataset(
                    mode='loc',
                    tracklet=t['stack_imgs'], 
                    args=args,
                    bbox=t['boxes']
                )

                pbar.set_description(f"Running T2 (number of tubes - action: {len(action_dataset)}, loc: {len(loc_dataset)})")
                
                # predict
                action_cls = []
                for tracklet in action_dataset:
                    input = torch.unsqueeze(tracklet, 0).to(int(args.devices))
                    pred = args.action_detector(input)
                    cls = torch.argmax(pred, dim=1)
                    action_cls.append(cls.item())

                loc_cls = []
                for stack_img, bbox in loc_dataset:
                    input = torch.unsqueeze(stack_img, 0).to(int(args.devices))
                    bbox = torch.unsqueeze(bbox, 0).to(int(args.devices))
                    pred = args.loc_detector(input, bbox)
                    cls = torch.argmax(pred, dim=1)
                    loc_cls.append(cls.item())

                # Padding and Matching t1 & t2 tubes
                event_tubes_list = event_tubes_list + make_t2_tube(t, action_cls, loc_cls)
    
    # bugs
    # for i in range(len(event_tubes_list)):
    #     event_tubes_list[i] = tube_interpolation(event_tubes_list[i])

    args.tube['triplet'][args.video_name] = event_tubes_list
    
    for i in range(len(args.tube['agent'])):
        args.tube['agent'][args.video_name][i] = tube_interpolation(args.tube['agent'][args.video_name][i])

    return 0


def merge_two_tube(args, major_tube, rare_tube):
    """
    ToDo: Merge tube using IoU.

    """
    for tube in rare_tube:
        tube['label_id'] += 2

    merged_tube = major_tube + rare_tube
    
    return merged_tube
    

def two_branch_yolo(args, video):
    """
    two branch tube pipeline
    
    Args:
        video: video path.
    """

    args.tracker = args.major_yolo.track(
        source=video,
        imgsz=args.imgsz,
        device=args.devices,
        stream=True,
        conf=0.0,
        verbose=False
    )
    print("major tracking")
    major_tube = make_tube(args)

    args.tracker = args.rare_yolo.track(
        source=video,
        imgsz=args.imgsz,
        device=args.devices,
        stream=True,
        conf=0.0,
        verbose=False
    )
    print("rare tracking")
    rare_tube = make_tube(args)

    args.tube['agent'][args.video_name] = merge_two_tube(args, major_tube, rare_tube)

    return 0

def main(args):
    """
        Args: see utils/opt.py
    """
    
    if args.mode == 'Track2':
        args.tube = {
            'agent': {},
            'triplet': {}
        }
    else:
        args.tube = {
            'agent': {}
        }
    all_vs = sorted(glob.glob(os.path.join(args.video_path, '*.mp4')))
    for idx, v in enumerate(tqdm(all_vs)):
        args.video_name = v.split('/')[-1].split('.')[0]
        
        if args.two_branch:
            two_branch_yolo(args, v)
        else:
            # tracking Using BoT-SORT
            args.tracker = args.yolo.track(
                source=v,
                imgsz=args.imgsz,
                device=args.devices,
                stream=True,
                conf = 0.0,
                verbose=False
            )

            make_tube(args)

        # ToDo: two branch T2
        if args.mode == 'Track2':
            track2(args)
            
        # # debug for one video
        # with open(args.pkl_name, 'wb') as f:
        #     pickle.dump(args.tube, f)
    
    tube_file_name = Path(args.pkl_dir)/f"{args.pkl_dir.parts[-1]}_tubes.pkl"
    print(f"writing {tube_file_name} ..")
    with open(tube_file_name, 'wb+') as f:
        pickle.dump(args.tube, f)
    print("zipping ..")
    zip_tube_file(file_path=tube_file_name)
  

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

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='Track1', help='detect mode, only accept Track1 or Track2')
    parser.add_argument('--video_path', type=str, default='./roadpp/test_videos', help='video path')
    parser.add_argument('--detector', type=str, default="yolo")
    parser.add_argument('--model_path', type=str, default='runs/detect/yolov8l_T1_1280_batch_8_/weights/best.pt', help='yolo path')


    parser.add_argument('--two_branch', type=bool, default=False, help='used two branch YOLO')
    parser.add_argument('--major_path', type=str, default='runs/detect/yolov8l_T1_1280_batch_8_/weights/best.pt', help='major_yolo path')
    parser.add_argument('--rare_path', type=str, default='runs/detect/yolov8l_T1_1280_batch_8_/weights/best.pt', help='rare_yolo path')

    parser.add_argument('--devices', nargs='+', type=str, default='0', help='gpu number')

    parser.add_argument('--imgsz', type=tuple, default=(1280, 1280), help='yolo input size')
    parser.add_argument('--video_shape', type=tuple, default=(1280, 1920), help='original video resolution')
    parser.add_argument('--submit_shape', type=tuple, default=(600, 840), help='final submit shape')

    parser.add_argument('--pkl_dir', type=Path, default=Path("./roadpp/submit"), help='submit file name(*.pkl)')
 
    # track2
    parser.add_argument('--action_detector_path', type=str, default='runs/action/best_weight.pt', help='action_detector_path')
    parser.add_argument('--loc_detector_path', type=str, default='runs/location/best_weight.pt', help='loc_detector_path')

    parser.add_argument('--t2_input_shape', type=tuple, default=(224, 224), help='t2_input_shape')
    parser.add_argument('--windows_size', type=int, default=4, help='sliding windows shape')
    

    opt = parser.parse_args()
    print(opt)
    return opt

if __name__ == '__main__':
    
    args = arg_parse()
    assert args.mode == 'Track1' or args.mode == 'Track2',\
        'detect mode only accept "Track1" or "Track2".'
    
    args.pkl_dir.mkdir(parents=True, exist_ok=True)
    
    detector = {"yolo":YOLO,"rtdetr":RTDETR} 

    print(args.detector)
    if args.two_branch:
        shutil.copy(args.major_path, args.pkl_dir/f"{args.detector}_major.pt")
        args.major_yolo = detector[args.detector](args.major_path)
        shutil.copy(args.rare_path, args.pkl_dir/f"{args.detector}_rare.pt")
        args.rare_yolo = detector[args.detector](args.rare_path)
        args.imgsz = 1920
    
    else:
        copy_model = args.pkl_dir/f'{args.detector}.pt'
        print(f"copy {args.model_path} to {copy_model}")
        shutil.copy(args.model_path, copy_model)
        args.yolo = detector[args.detector](copy_model)
    
    if args.mode == 'Track2':
        args.action_detector = torch.load(args.action_detector_path)
        args.action_detector.eval()

        args.loc_detector = torch.load(args.loc_detector_path)
        args.loc_detector.eval()
    
    check_cuda()
    main(args)
