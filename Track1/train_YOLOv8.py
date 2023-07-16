from ultralytics import YOLO

    
if __name__ == '__main__':
    # Parameter
    pretrain_weight = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/Pretrain/yolov8l.pt'
    imgsz = [1280, 1920]
    batch_size = 26
    name = 'yolov8l_T1_' + str(imgsz[1]) + '_batch_' + str(batch_size) + '_'

    # Training.
    model = YOLO(model=pretrain_weight)

    results = model.train(
    data = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/Track1/track1.yaml',
    imgsz = 1280,
    device = 0,
    epochs = 200,
    batch = 8,
    name = 'yolov8l_T1_1280_batch_8'
    )