import numpy as np
from wildlife_tools.similarity.wildfusion import WildFusion


def best_matches_matching(results, max_per_encounter, min_similarity=0):

    if results["encounter1"].isnull().any():
        raise ValueError("nan in encounter1 encountered")

    # Sort the similarity and keep only one match between unique images (bounding boxes)
    results = results[results["similarity"] > min_similarity]
    results = results.sort_values("similarity", ascending=False)
    results = results.reset_index(drop=True)
    idx_keep = results[["image_id1", "image_id2"]].drop_duplicates().index
    results = results.loc[idx_keep]
    results = results.reset_index(drop=True)

    # Keep only at most max_per_identity matches per a par of identities
    idx_keep = []
    for _, result_encounter in results.groupby("encounter1"):
        idx_keep = idx_keep + list(result_encounter.index[:max_per_encounter])
    results = results.loc[idx_keep]
    return results.reset_index(drop=True)


def best_matches_discovery(
    results, max_matches, max_per_identity, unlimited_unknown=True, min_similarity=0
):

    # Sort the similarity and keep only one match between unique images (bounding boxes)
    results = results[results["similarity"] > min_similarity]
    results = results.sort_values("similarity", ascending=False)
    results = results.reset_index(drop=True)
    idx_keep = results[["image_id1", "image_id2"]].drop_duplicates().index
    results = results.loc[idx_keep]
    results = results.reset_index(drop=True)

    # Keep only at most max_per_identity matches per a par of identities
    results["key"] = results.apply(
        lambda r: frozenset((r["identity1"], r["identity2"])), axis=1
    )
    idx_remove = []
    for identities, result_identity in results.groupby("key", sort=False):
        if unlimited_unknown and "unknown" in identities:
            continue
        idx_remove = idx_remove + list(result_identity.index[max_per_identity:])
    idx_keep = np.setdiff1d(range(len(results)), idx_remove)
    results = results.loc[idx_keep]

    # Keep only at most max_matches matches
    results = results[:max_matches]
    return results.reset_index(drop=True)


def compute_similarity(query, database, ignore_pairs, priority, pipelines, Bs, names):
    similarities = {}
    for pipeline, B, name in zip(pipelines, Bs, names):
        wildfusion = WildFusion(
            calibrated_pipelines=[pipeline], priority_pipeline=priority
        )
        similarity = wildfusion(query, database, B=B, ignore_pairs=ignore_pairs)
        similarity = similarity.astype(np.float64)
        similarities[name] = similarity
    return similarities


def top_indices(similarity, ignore_lower_diagonal=False):
    if ignore_lower_diagonal:
        similarity[np.tril_indices_from(similarity)] = -np.inf
    flat_arr = similarity.flatten()
    top_flat_indices = np.argsort(-flat_arr)
    return np.unravel_index(top_flat_indices, similarity.shape)
