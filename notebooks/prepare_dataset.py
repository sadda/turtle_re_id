from animal_re_id.models import run_detection, turtle_detector
from animal_re_id.turtlewatch_egypt import TWE_NewData

roots = [
    '/data/wildlife_datasets/turtle_watch/2017 PICS + DATASHEET',
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
    TWE_NewData.get_data(root, force=True)
    dataset = TWE_NewData(root)    
    segmentation = run_detection(dataset, model)
    segmentation['image_id'] = segmentation['image_id'].astype(int)
    segmentation.to_csv(f'{root}/segmentation.csv', index=False)
