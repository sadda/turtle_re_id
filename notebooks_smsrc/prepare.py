from animal_re_id import run_detection, turtle_detector
from wildlife_datasets.datasets import TurtlesOfSMSRC

root = '/data/wildlife_datasets/TurtlesOfSMSRC'
TurtlesOfSMSRC.get_data(root, force=True)
dataset = TurtlesOfSMSRC(root)

model = turtle_detector()
segmentation = run_detection(dataset, model)
segmentation.to_csv(f'{root}/segmentation.csv', index=False)