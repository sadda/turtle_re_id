from animal_re_id.models import run_detection, turtle_detector
from wildlife_datasets.datasets import TurtlewatchEgypt_New, utils

delete_corrupted = False
file_name = "animal_re_id/turtlewatch_egypt/individuals.csv"
roots = [
    # '/data/wildlife_datasets/turtle_watch/2017 PICS + DATASHEET',
    '/data/wildlife_datasets/turtle_watch/2018 PICS + DATASHEET',
    '/data/wildlife_datasets/turtle_watch/2019 PICS + DATASHEET',
    '/data/wildlife_datasets/turtle_watch/2020 PICS + DATASHEET',
    '/data/wildlife_datasets/turtle_watch/2021 PICS + DATASHEET',
    '/data/wildlife_datasets/turtle_watch/2022 PICS + online database',
    '/data/wildlife_datasets/turtle_watch/2023 PICS + online database',
    '/data/wildlife_datasets/turtle_watch/2024 PICS + online database',
    '/data/wildlife_datasets/turtle_watch/2025 PICS + online database',    
] 
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
    segmentation.to_csv(f'{root}/segmentation.csv', index=False)
