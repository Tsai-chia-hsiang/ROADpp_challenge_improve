import pickle
import os
from pathlib import Path
import os.path as osp
import numpy as np
import zipfile

def switch_action_labels(action_id):
    if action_id == 5 or action_id == 6 or action_id == 16:
        return 3
    elif action_id == 20:
        return 19
        
    return action_id


def switch_loc_labels(loc_id):
    if loc_id == 3:
        return 2
    elif loc_id == 5 or loc_id == 6:
        return 4
    elif loc_id == 14 or loc_id == 15:
        return 13
    
    return loc_id


def combine_label(agent_id, action_id, loc_id):
    agent_labels = ['Ped', 'Car', 'Cyc', 'Mobike', 'SmalVeh', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL']
    action_labels = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Rev', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'MovRht', 'MovLft', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj']
    loc_labels = ['VehLane', 'OutgoLane', 'OutgoCycLane', 'OutgoBusLane', 'IncomLane', 'IncomCycLane', 'IncomBusLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking', 'LftParking', 'rightParking']
    triplet_labels = ['Ped-MovAway-LftPav', 'Ped-MovAway-RhtPav', 'Ped-MovAway-Jun', 'Ped-MovTow-LftPav', 'Ped-MovTow-RhtPav', 'Ped-MovTow-Jun', 'Ped-Mov-OutgoLane', 'Ped-Mov-Pav', 'Ped-Mov-RhtPav', 'Ped-Stop-OutgoLane', 'Ped-Stop-Pav', 'Ped-Stop-LftPav', 'Ped-Stop-RhtPav', 'Ped-Stop-BusStop', 'Ped-Wait2X-RhtPav', 'Ped-Wait2X-Jun', 'Ped-XingFmLft-Jun', 'Ped-XingFmRht-Jun', 'Ped-XingFmRht-xing', 'Ped-Xing-Jun', 'Ped-PushObj-LftPav', 'Ped-PushObj-RhtPav', 'Car-MovAway-VehLane', 'Car-MovAway-OutgoLane', 'Car-MovAway-Jun', 'Car-MovTow-VehLane', 'Car-MovTow-IncomLane', 'Car-MovTow-Jun', 'Car-Brake-VehLane', 'Car-Brake-OutgoLane', 'Car-Brake-Jun', 'Car-Stop-VehLane', 'Car-Stop-OutgoLane', 'Car-Stop-IncomLane', 'Car-Stop-Jun', 'Car-Stop-parking', 'Car-IncatLft-VehLane', 'Car-IncatLft-OutgoLane', 'Car-IncatLft-IncomLane', 'Car-IncatLft-Jun', 'Car-IncatRht-VehLane', 'Car-IncatRht-OutgoLane', 'Car-IncatRht-IncomLane', 'Car-IncatRht-Jun', 'Car-HazLit-IncomLane', 'Car-TurLft-VehLane', 'Car-TurLft-Jun', 'Car-TurRht-Jun', 'Car-MovRht-OutgoLane', 'Car-MovLft-VehLane', 'Car-MovLft-OutgoLane', 'Car-XingFmLft-Jun', 'Car-XingFmRht-Jun', 'Cyc-MovAway-OutgoCycLane', 'Cyc-MovAway-RhtPav', 'Cyc-MovTow-IncomLane', 'Cyc-MovTow-RhtPav', 'MedVeh-MovAway-VehLane', 'MedVeh-MovAway-OutgoLane', 'MedVeh-MovAway-Jun', 'MedVeh-MovTow-IncomLane', 'MedVeh-MovTow-Jun', 'MedVeh-Brake-VehLane', 'MedVeh-Brake-OutgoLane', 'MedVeh-Brake-Jun', 'MedVeh-Stop-VehLane', 'MedVeh-Stop-OutgoLane', 'MedVeh-Stop-IncomLane', 'MedVeh-Stop-Jun', 'MedVeh-Stop-parking', 'MedVeh-IncatLft-IncomLane', 'MedVeh-IncatRht-Jun', 'MedVeh-TurRht-Jun', 'MedVeh-XingFmLft-Jun', 'MedVeh-XingFmRht-Jun', 'LarVeh-MovAway-VehLane', 'LarVeh-MovTow-IncomLane', 'LarVeh-Stop-VehLane', 'LarVeh-Stop-Jun', 'Bus-MovAway-OutgoLane', 'Bus-MovTow-IncomLane', 'Bus-Stop-VehLane', 'Bus-Stop-OutgoLane', 'Bus-Stop-IncomLane', 'Bus-Stop-Jun', 'Bus-HazLit-OutgoLane']

    # event_action_labels = [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21]
    # event_loc_labels = [0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13]
    
    # assign cls which didn't appear in triplet labels into some cls
    switched_action_id = switch_action_labels(action_id)
    switched_loc_id = switch_loc_labels(loc_id)
    
    tube_tri_label = agent_labels[agent_id] + '-' + action_labels[switched_action_id] + '-' + loc_labels[switched_loc_id]

    if tube_tri_label in triplet_labels:
        return triplet_labels.index(tube_tri_label)
    else:
        return -1
    
    
def bbox_normalized(bbox, img_w, img_h):
    return bbox / np.array([img_w, img_h, img_w, img_h])


def norm_box_into_absolute(bbox, img_w, img_h):
    return bbox * np.array([img_w, img_h, img_w, img_h])


def tube_change_axis(tube, orig_shape, submit_shape):
    ori_h, ori_w = orig_shape
    new_h, new_w = submit_shape
    
    tube['boxes'] = np.array([norm_box_into_absolute(bbox_normalized(box, ori_w, ori_h), new_w, new_h) for box in tube['boxes']])
    
    return tube


def pkl_change_axis(tubes, ori_w, ori_h, new_w, new_h):
    for video, tube in tubes.items():
        for t in tube:
            t['boxes'] = np.array([norm_box_into_absolute(bbox_normalized(box, ori_w, ori_h), new_w, new_h) for box in t['boxes']])
            
    return tubes


def action_tube_padding(action_cls, prev_frames, last_frames, frames_len):
    if frames_len < 4:
        return action_cls + [action_cls[0]] * (frames_len - 1)
    else:
        padded_front = [action_cls[0]] * prev_frames + action_cls
        padded_full = padded_front + [action_cls[-1]] * last_frames
        
        return padded_full


def stack_imgs_padding(stack_imgs):
    if len(stack_imgs) == 1:
        return stack_imgs + ([stack_imgs[0]] * 3)
    elif len(stack_imgs) == 2:
        return [stack_imgs[0]] + stack_imgs + [stack_imgs[1]]
    elif len(stack_imgs) == 3:
        return stack_imgs + [stack_imgs[-1]]
    else:
        return stack_imgs    


def zip_tube_file(file_path:Path):

    zip_path = file_path.parent/f"{file_path.stem}.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add the file to the zip archive
        zipf.write(file_path,file_path.name)


if __name__ == '__main__':
    zip_tube_file(Path("../roadpp/submit_file/t1_rtdetr_v3.pkl"))