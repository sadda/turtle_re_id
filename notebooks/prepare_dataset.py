from wildlife_datasets.datasets import (
    TurtlewatchEgypt_Master,
    TurtlewatchEgypt_New,
    utils,
)
from wildlife_datasets.detection import run_detection

from animal_re_id.config import load_paths, resolve_path
from animal_re_id.models import turtle_detector

delete_corrupted = False
paths = load_paths("TurtlewatchEgypt")
file_name = resolve_path(paths["file_name"])
root_new = resolve_path(paths["root_new"])
roots = [f"{root_new}/{r}" for r in paths["roots_new"]]
root_master = resolve_path(paths["root_heads"])
model = turtle_detector()

for root in roots:
    # Load dataset
    dataset = TurtlewatchEgypt_New(root, file_name=file_name)

    # Delete corrupted files
    if delete_corrupted:
        corrupted = utils.find_corrupted_images(root)
        utils.delete_corrupted_images(root, corrupted)

    # Extract segmentation
    segmentation = run_detection(dataset, model)
    segmentation.to_csv(f"{root}/segmentation.csv", index=False)

# Do the same for master database of heads
dataset = TurtlewatchEgypt_Master(root_master, file_name=file_name)
segmentation = run_detection(dataset, model)
segmentation.to_csv(f"{root_master}/segmentation.csv", index=False)
