import numpy as np
import os
import pandas as pd
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from tqdm import tqdm

def run_detection(dataset, model):
    image_ids = np.empty(0, dtype=int)
    bboxes = np.empty((0,4))
    scores = np.empty(0, dtype=float)
    labels = np.empty(0, dtype=int)
    for _, df_row in tqdm(dataset.df.iterrows(), total=len(dataset)):
        file_name = os.path.join(dataset.root, df_row['path'])        
        try:
            result = model.predict(source=file_name, verbose=False, save=False, show=False)[0]
        except:
            continue

        for box in result.boxes:
            image_ids = np.concatenate((image_ids, [df_row['image_id']]))
            bboxes = np.concatenate((bboxes, box.xyxy.cpu().numpy()))
            scores = np.concatenate((scores, box.conf.cpu().numpy()))
            labels = np.concatenate((labels, box.cls.cpu().numpy()))

    return pd.DataFrame({
        'image_id': image_ids,
        'bbox_x': bboxes[:,0],
        'bbox_y': bboxes[:,1],
        'bbox_w': bboxes[:,2]-bboxes[:,0],
        'bbox_h': bboxes[:,3]-bboxes[:,1],
        'score': scores,
        'label': [result.names[x] for x in labels]
    })

def turtle_detector(
        repo_id ="BVRA/TurtleDetector",
        filename = "turtle_detector.pt",
        ):

    path_model = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(path_model)