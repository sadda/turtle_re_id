import cv2
import numpy as np
import os
import pandas as pd
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from tqdm import tqdm
from pycocotools import mask as mask_utils
from wildlife_datasets.datasets import WildlifeDataset

def turtle_detector(
        repo_id: str ="BVRA/TurtleDetector",
        filename: str = "turtle_detector.pt",
        ) -> YOLO:

    path_model = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(path_model)