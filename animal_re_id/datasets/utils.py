from datetime import datetime
from pathlib import Path

import pandas as pd
from wildlife_datasets.datasets import WildlifeDataset

MIN_IMAGE_ID_LENGTH = 10


def add_date_exif(dataset: WildlifeDataset, i: int, file_name: str) -> None:
    date = datetime.fromisoformat(dataset.df.loc[i, "date"])
    date = date.strftime("%Y:%m:%d %H:%M:%S")

    img = dataset[i]
    exif = img.getexif()

    for tag in [306, 36867, 36868]:
        exif[tag] = date
    img.save(file_name, exif=exif)


def check_image_ids(*metadatas: pd.DataFrame) -> None:
    for metadata in metadatas:
        image_ids = metadata["image_id"]
        is_string = image_ids.apply(lambda x: isinstance(x, str))
        is_short = image_ids.astype(str).str.len() < MIN_IMAGE_ID_LENGTH
        wrong = image_ids[~is_string | is_short]
        if len(wrong) > 0:
            raise ValueError(
                f"image_id must be strings with at least {MIN_IMAGE_ID_LENGTH} characters to avoid collisions in the cache (found e.g. {wrong.iloc[0]}). "
                'Create them from the paths with wildlife_datasets.datasets.utils.get_persistent_id(metadata["path"]).'
            )

    # Different datasets must not share any image_id, otherwise they share the cache
    for i, metadata1 in enumerate(metadatas):
        for j, metadata2 in enumerate(metadatas[i + 1 :], start=i + 1):
            if metadata1.equals(metadata2):
                continue
            shared = set(metadata1["image_id"]) & set(metadata2["image_id"])
            if len(shared) > 0:
                raise ValueError(f"Datasets {i} and {j} share image_id, which would mix them in the cache (found e.g. {next(iter(shared))}).")


def show_tree(path: Path, max_items: int | None = 10, indent: int = 0) -> None:
    ignore_folders = (".DS_Store",)
    items = []
    for item in sorted(path.iterdir()):
        if item.name not in ignore_folders:
            items.append(item)
    shown = items if max_items is None else items[:max_items]
    for item in shown:
        print("  " * indent + ("📁 " if item.is_dir() else "📄 ") + item.name)
        if item.is_dir() and not item.is_symlink():
            show_tree(item, max_items, indent + 1)
    if len(items) > len(shown):
        print("  " * indent + f"... ({len(items) - len(shown)} more)")
