from wildlife_datasets.datasets import (
    TurtlewatchEgypt_Master,
    TurtlewatchEgypt_New,
    utils,
)
from wildlife_datasets.detection import run_detection

from animal_re_id.config import load_paths, resolve_path
from animal_re_id.models import turtle_detector

delete_corrupted = False
check_files = False
paths = load_paths()
file_name = resolve_path(paths["twe"]["file_name"])
root_new = resolve_path(paths["twe"]["root_new"])
roots = [f"{root_new}/{r}" for r in paths["twe"]["roots_new"]]
root_master = resolve_path(paths["twe"]["root_heads"])
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
