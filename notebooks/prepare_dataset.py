from wildlife_datasets.datasets import (
    TurtlewatchEgypt_Master,
    TurtlewatchEgypt_New,
    utils,
)
from wildlife_datasets.detection import run_detection

from animal_re_id.models import turtle_detector

delete_corrupted = False
check_files = False
file_name = "individuals.csv"
roots = [
    "/data/wildlife_datasets/turtle_watch/2017 PICS + DATASHEET",
    "/data/wildlife_datasets/turtle_watch/2018 PICS + DATASHEET",
    "/data/wildlife_datasets/turtle_watch/2019 PICS + DATASHEET",
    "/data/wildlife_datasets/turtle_watch/2020 PICS + DATASHEET",
    "/data/wildlife_datasets/turtle_watch/2021 PICS + DATASHEET",
    "/data/wildlife_datasets/turtle_watch/2022 PICS + online database",
    "/data/wildlife_datasets/turtle_watch/2023 PICS + online database",
    "/data/wildlife_datasets/turtle_watch/2024 PICS + online database",
    "/data/wildlife_datasets/turtle_watch/2025 PICS + online database",
]
root_master = "/data/wildlife_datasets/turtle_watch/PROFILES"
model = turtle_detector()

for root in roots:
    # Load dataset
    dataset = TurtlewatchEgypt_New(root, file_name=file_name, check_files=check_files)

    # Delete corrupted files
    if delete_corrupted:
        corrupted = utils.find_corrupted_images(root)
        utils.delete_corrupted_images(root, corrupted)

    # Extract segmentation
    segmentation = run_detection(dataset, model)
    segmentation.to_csv(f"{root}/segmentation.csv", index=False)

# Do the same for master database of heads
root = root_master
dataset = TurtlewatchEgypt_Master(root, file_name=file_name, check_files=check_files)
segmentation = run_detection(dataset, model)
segmentation.to_csv(f"{root}/segmentation.csv", index=False)
