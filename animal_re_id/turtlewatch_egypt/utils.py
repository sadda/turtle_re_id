import os
import requests

def download_detector(
        url = 'https://huggingface.co/BVRA/TurtleDetector/resolve/main/turtle_detector.pt',
        path_model = 'models/turtle_detector.pt',
        force = False,
        ):

    if not os.path.exists(path_model) or force:
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        response = requests.get(url)
        with open(path_model, 'wb') as f:
            f.write(response.content)
