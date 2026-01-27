import os
import requests

def strip_suffixes(x, suffixes):
    suffixes = suffixes + [x.lower() for x in suffixes]
    while any(x.endswith(ext) for ext in suffixes):
        for ext in suffixes:
            if x.endswith(ext):
                x = x.removesuffix(ext).strip()
                break
    return x

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
