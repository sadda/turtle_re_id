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
