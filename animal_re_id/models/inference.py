import numpy as np
from wildlife_tools.similarity.wildfusion import WildFusion

def compute_similarity(query, database, ignore_pairs, priority, pipelines, Bs, names):
    similarities = {}
    for pipeline, B, name in zip(pipelines, Bs, names):
        wildfusion = WildFusion(calibrated_pipelines = [pipeline], priority_pipeline = priority)
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

