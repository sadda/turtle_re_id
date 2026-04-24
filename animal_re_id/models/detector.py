from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def turtle_detector(
    repo_id: str = "BVRA/TurtleDetector",
    filename: str = "turtle_detector.pt",
) -> YOLO:

    path_model = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(path_model)
