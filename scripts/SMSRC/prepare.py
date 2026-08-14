from wildlife_datasets.datasets import TurtlesOfSMSRC
from wildlife_datasets.detection import run_detection

from animal_re_id.models import turtle_detector

root = "/data/wildlife_datasets/data/TurtlesOfSMSRC"
TurtlesOfSMSRC.get_data(root, force=True)
dataset = TurtlesOfSMSRC(root)

model = turtle_detector()
segmentation = run_detection(dataset, model)
segmentation.to_csv(f"{root}/segmentation.csv", index=False)
