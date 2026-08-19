from wildlife_tools.similarity import CollectAll, MatchLightGlue
from wildlife_tools.similarity.pairwise.utils import visualise_matches


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


def visualise_match(dataset_q, dataset_d, result, pipeline, features_name, init_threshold=0.65, ax=None):

    img_q, features_q = extract_features(dataset_q, pipeline, result["image_id1"])
    img_d, features_d = extract_features(dataset_d, pipeline, result["image_id2"])

    matcher = MatchLightGlue(features=features_name, collector=CollectAll(), init_threshold=init_threshold)
    results = matcher(features_q, features_d)[0]
    visualise_matches(img_q, results["kpts0"], img_d, results["kpts1"], ax=ax)
