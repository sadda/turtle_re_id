import os

import pandas as pd
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_datasets.detection import run_detection


def turtle_detector(
    repo_id: str = "BVRA/TurtleDetector",
    filename: str = "turtle_detector.pt",
) -> YOLO:

    path_model = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(path_model)


def run_detection_cached(dataset: WildlifeDataset, model: YOLO, file_name: str) -> pd.DataFrame:
    image_ids = dataset.metadata["image_id"]

    # Keep previous detections of images which are still in the dataset
    if os.path.exists(file_name):
        segmentation = pd.read_csv(file_name, dtype={"image_id": image_ids.dtype}, float_precision="round_trip")
        segmentation = segmentation[segmentation["image_id"].isin(image_ids)]
    else:
        segmentation = pd.DataFrame({"image_id": pd.Series(dtype=image_ids.dtype)})

    # Run detection only on images without previous detections
    mask = ~image_ids.isin(segmentation["image_id"])
    if mask.any():
        dataset_new = dataset.get_subset(mask)
        segmentation_new = run_detection(dataset_new, model)

        # Store images without detections as rows with nans so that they are not processed again
        image_ids_new = dataset_new.metadata["image_id"]
        image_ids_empty = image_ids_new[~image_ids_new.isin(segmentation_new["image_id"])]
        segmentation_empty = pd.DataFrame({"image_id": image_ids_empty})
        segmentation = pd.concat([segmentation, segmentation_new, segmentation_empty], ignore_index=True)

    # Sort detections by the order of images in the dataset
    position = dict(zip(image_ids, range(len(image_ids))))
    order = segmentation["image_id"].map(position)
    segmentation = segmentation.iloc[order.argsort(kind="stable")].reset_index(drop=True)
    segmentation.to_csv(file_name, index=False)
    return segmentation
