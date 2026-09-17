import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from tqdm import tqdm
from wildlife_datasets.datasets import StripeSpotter


def segment_image(processor, image: Image.Image, prompt="animal") -> Image.Image | None:
    with torch.autocast(device_type="cuda:2", dtype=torch.bfloat16):
        inference_state = processor.set_image(image)

        # Process text prompt
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        # Get the masks, bounding boxes, and scores
        masks, _, _ = output["masks"], output["boxes"], output["scores"]
        if len(masks) == 0:
            return None

        # Select the biggest mask
        masks_sum = [mask.sum().detach().cpu() for mask in masks]
        i = np.argmax(masks_sum)
        mask = masks[i][0].cpu().numpy()
        img = np.asarray(image)

        return Image.fromarray(mask[..., None] * img)


def save_image(path: str, image: Image.Image) -> None:
    base_dir = os.path.dirname(path)
    os.makedirs(base_dir, exist_ok=True)

    if path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
        image.save(path, format="JPEG", subsampling=0, quality=100)
    else:
        image.save(path)


root = "/data/wildlife_datasets/data/StripeSpotter"
destination_directory = "/data/wildlife_datasets/data/StripeSpotter_segmented"


model = build_sam3_image_model()
processor = Sam3Processor(model)


dataset = StripeSpotter(root)
for i in tqdm(range(len(dataset))):
    path = dataset.df.iloc[i]["path"]
    path_new = os.path.join(destination_directory, path)
    if os.path.exists(path_new):
        continue

    image = dataset[i]
    assert isinstance(image, Image.Image)
    seg_image = segment_image(processor, image)

    if seg_image is None:
        seg_image = segment_image(processor, image, prompt="zebra")
        seg_image = seg_image if seg_image is not None else image

    save_image(path_new, seg_image)


idx = [0, 2, 5, 8, 9, 10]
dataset = StripeSpotter(destination_directory)
dataset.plot_grid(idx=idx, rotate=False, n_cols=3, n_rows=2, img_min=300)
plt.savefig("StripeSpotter_full.png", dpi=300, bbox_inches="tight")
dataset = StripeSpotter(destination_directory, img_load="bbox")
dataset.plot_grid(idx=idx, rotate=False, n_cols=3, n_rows=2, img_min=300)
plt.savefig("StripeSpotter_flanks.png", dpi=300, bbox_inches="tight")
