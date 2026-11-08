import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib.axes import Axes
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.data import FeatureDataset
from wildlife_tools.similarity import CollectAll, MatchLightGlue, SimilarityPipeline
from wildlife_tools.similarity.pairwise.utils import visualise_matches as wt_visualise_matches


def extract_features(dataset: WildlifeDataset, pipeline: SimilarityPipeline, i: str) -> tuple[Image.Image, FeatureDataset]:
    j = get_index(dataset, i)
    dataset = dataset.get_subset([j])
    with dataset.temporary_attrs(transform=pipeline.transform.transforms[0]):
        img, _ = dataset[0]
    with dataset.temporary_attrs(transform=pipeline.transform):
        features = pipeline.extractor(dataset)
    return img, features


def get_index(dataset: WildlifeDataset, image_id: str) -> int | None:
    idx = dataset.metadata["image_id"] == image_id
    if idx.sum() > 1:
        raise ValueError("image_id found multiple times.")
    elif idx.sum() == 1:
        return dataset.metadata[idx].index[0]
    elif idx.sum() == 0:
        print(f"image_id {image_id} not found")
        return None


def plot_matches(dataset: WildlifeDataset, result: pd.Series) -> None:
    i_q = get_index(dataset, result["image_id1"])
    i_d = get_index(dataset, result["image_id2"])
    identities = result[["identity1", "identity2"]].to_numpy()
    header_cols = [f"{identities[0]}_{identities[1]}", result["similarity"]]
    dataset.plot_grid(idx=[i_q, i_d], n_cols=2, header_cols=header_cols, rotate=False)


def visualise_matches(query: WildlifeDataset, database: WildlifeDataset, row: pd.Series, pipeline: SimilarityPipeline, features_name: str, init_threshold: float = 0.5, ax: Axes | None = None, title: str | None = None) -> Axes:
    img_q, features_q = extract_features(query, pipeline, row["image_id1"])
    img_d, features_d = extract_features(database, pipeline, row["image_id2"])

    matcher = MatchLightGlue(features=features_name, collector=CollectAll(), init_threshold=init_threshold)
    results = matcher(features_q, features_d)[0]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    wt_visualise_matches(img_q, results["kpts0"], img_d, results["kpts1"], ax=ax)

    if title is not None:
        ax.set_title(title)
    return ax
