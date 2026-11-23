import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.similarity import CollectAll, MatchLightGlue

from animal_re_id.config import load_paths, resolve_path
from animal_re_id.models import get_pipeline_aliked, turtle_detector
from animal_re_id.models.utils import extract_features

photos = {
    "2019": "2019 PICS + DATASHEET/10October2019/MADA/05-10-2019/SIGHTING# 4/IMG_8077.JPG",
    "2023": "2023 PICS + online database/02 February 2023/MADA/2023_02_25/SIGHTING# 7. KIKI/GOPR3726.JPG",
}
parts = {
    "head": ["head"],
    "front flipper": ["flipper_fl", "flipper_fr"],
    "hind flipper": ["flipper_hl", "flipper_hr"],
}
device = "cuda"
init_threshold = 0.5
max_photo_size = 1920

local_path = "config/paths.json"
paths = load_paths("TurtlewatchEgypt", local_path=local_path)
root_new = resolve_path(paths["root_new"])
root_assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "kiki")
os.makedirs(root_assets, exist_ok=True)


def save_photo(path: str, key: str, model) -> list[dict]:
    result = model.predict(source=f"{root_new}/{path}", verbose=False)[0]
    img = Image.fromarray(result.orig_img[:, :, ::-1])
    scale = min(1.0, max_photo_size / max(img.size))
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    img.save(f"{root_assets}/photo_{key}.jpg", quality=92)

    detections = []
    for box, polygon in zip(result.boxes, result.masks.xy):
        label = result.names[int(box.cls)]
        if label == "turtle":
            continue
        x0, y0, x1, y1 = (box.xyxy[0].cpu().numpy() * scale).tolist()
        polygon = (np.asarray(polygon) * scale).round(1).tolist()
        detections.append({"label": label, "score": float(box.conf), "bbox": [x0, y0, x1 - x0, y1 - y0], "polygon": polygon})
    return detections


def match_part(dataset: WildlifeDataset, pipeline, matcher: MatchLightGlue, labels: list[str], part: str) -> dict:
    metadata = dataset.metadata
    best = None
    for label in labels:
        ids1 = metadata.loc[(metadata["key"] == "2019") & (metadata["label"] == label), "image_id"]
        ids2 = metadata.loc[(metadata["key"] == "2023") & (metadata["label"] == label), "image_id"]
        for id1 in ids1:
            for id2 in ids2:
                img1, features1 = extract_features(dataset, pipeline, id1)
                img2, features2 = extract_features(dataset, pipeline, id2)
                result = matcher(features1, features2)[0]
                n = len(result["kpts0"])
                if best is None or n > best["n"]:
                    best = {"n": n, "id1": id1, "id2": id2, "size1": img1.size, "size2": img2.size, "result": result}
    if best is None:
        raise ValueError(f"{part} not detected in both photos")

    out = {"part": part}
    for k, image_id, size, kpts_key in [(1, best["id1"], best["size1"], "kpts0"), (2, best["id2"], best["size2"], "kpts1")]:
        row = metadata[metadata["image_id"] == image_id].iloc[0]
        x, y, w, h = row["bbox"]
        crop = Image.open(f"{root_assets}/photo_{row['key']}.jpg").crop((round(x), round(y), round(x + w), round(y + h)))
        file_name = f"crop_{part.replace(' ', '_')}_{k}.png"
        crop.save(f"{root_assets}/{file_name}")
        kpts = np.asarray(best["result"][kpts_key], dtype=float)
        kpts[:, 0] *= crop.width / size[0]
        kpts[:, 1] *= crop.height / size[1]
        out[f"crop{k}"] = file_name
        out[f"label{k}"] = row["label"]
        out[f"kpts{k}"] = kpts.round(1).tolist()
    out["n_matches"] = best["n"]
    return out


model = turtle_detector()
detections = {key: save_photo(path, key, model) for key, path in photos.items()}

rows = []
for key, dets in detections.items():
    for i, det in enumerate(dets):
        rows.append({
            "image_id": f"{key}_{i}",
            "identity": "kiki",
            "path": f"photo_{key}.jpg",
            "key": key,
            "label": det["label"],
            "bbox": det["bbox"],
        })
df = pd.DataFrame(rows)
dataset = WildlifeDataset(root=root_assets, df=df, img_load="bbox", load_label=True)

pipeline = get_pipeline_aliked(device=device)
matcher = MatchLightGlue(features="aliked", collector=CollectAll(), init_threshold=init_threshold)
matches = [match_part(dataset, pipeline, matcher, labels, part) for part, labels in parts.items()]

assets = {"photos": {key: f"photo_{key}.jpg" for key in photos}, "detections": detections, "matches": matches}
with open(f"{root_assets}/assets.json", "w") as file:
    json.dump(assets, file, indent=1)

for m in matches:
    print(m["part"], m["label1"], m["label2"], m["n_matches"])
