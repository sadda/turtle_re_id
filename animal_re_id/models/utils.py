import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.similarity import CollectAll, MatchLightGlue
from wildlife_tools.similarity.pairwise.utils import visualise_matches as wt_visualise_matches


def extract_features(dataset, pipeline, i):
    j = get_index(dataset, i)
    dataset = dataset.get_subset([j])
    dataset.transform = pipeline.transform.transforms[0]
    img, _ = dataset[0]
    dataset.transform = pipeline.transform
    features = pipeline.extractor(dataset)
    return img, features


def get_index(dataset, image_id):
    idx = dataset.metadata["image_id"] == image_id
    if idx.sum() > 1:
        raise ValueError("image_id found multiple times.")
    elif idx.sum() == 1:
        return dataset.metadata[idx].index[0]
    elif idx.sum() == 0:
        print(f"image_id {image_id} not found")
        return None


def plot_matches(dataset, result):
    i_q = get_index(dataset, result["image_id1"])
    i_d = get_index(dataset, result["image_id2"])
    identities = result[["identity1", "identity2"]].to_numpy()
    header_cols = [f"{identities[0]}_{identities[1]}", result["similarity"]]
    dataset.plot_grid(idx=[i_q, i_d], n_cols=2, header_cols=header_cols, rotate=False)


def visualise_matches(query: WildlifeDataset, database: WildlifeDataset, row: pd.Series, pipeline, features_name: str, init_threshold: float = 0.5, ax: Axes | None = None, title: str | None = None) -> Axes:
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
