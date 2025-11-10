import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from turtlewatch_egypt import TurtleWatchEgypt_NYED

path_model = 'models/segmentation.pt'
roots = [
    '/data/wildlife_datasets/turtle_watch/2018 PICS + DATASHEET',
    '/data/wildlife_datasets/turtle_watch/2019 PICS + DATASHEET',
    '/data/wildlife_datasets/turtle_watch/2020 PICS + DATASHEET',
    '/data/wildlife_datasets/turtle_watch/2021 PICS + DATASHEET',
    '/data/wildlife_datasets/turtle_watch/2022 PICS + online database',
    '/data/wildlife_datasets/turtle_watch/2023 PICS + online database',    
    '/data/wildlife_datasets/turtle_watch/2024 PICS + online database',
]
model = YOLO(path_model)

for root in roots:
    TurtleWatchEgypt_NYED.get_data(root, force=True)
    dataset = TurtleWatchEgypt_NYED(root)    

    image_ids = np.empty(0, dtype=int)
    bboxes = np.empty((0,4))
    scores = np.empty(0, dtype=float)
    labels = np.empty(0, dtype=int)
    for i, (_, df_row) in enumerate(tqdm(dataset.df.iterrows(), total=len(dataset))):
        file_name = os.path.join(root, df_row['path'])        
        try:
            result = model.predict(source=file_name, verbose=False, save=False, show=False)[0]
        except:
            continue

        for box in result.boxes:
            cls = box.cls[0].cpu().item()
            conf = box.conf[0].cpu().item()
            bbox = box.xywh[0].cpu().numpy()

            image_ids = np.concatenate((image_ids, [df_row['image_id']]))
            bboxes = np.concatenate((bboxes, box.xyxy.cpu().numpy()))
            scores = np.concatenate((scores, box.conf.cpu().numpy()))
            labels = np.concatenate((labels, box.cls.cpu().numpy()))

    segmentation = pd.DataFrame({
        'image_id': image_ids,
        'bbox_x': bboxes[:,0],
        'bbox_y': bboxes[:,1],
        'bbox_w': bboxes[:,2]-bboxes[:,0],
        'bbox_h': bboxes[:,3]-bboxes[:,1],
        'score': scores,
        'label': [result.names[x] for x in labels]
    })
    segmentation['image_id'] = segmentation['image_id'].astype(int)
    segmentation.to_csv(f'{root}/segmentation.csv', index=False)
