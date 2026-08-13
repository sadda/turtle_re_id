from abc import ABC, abstractmethod

import numpy as np


def _apply_ignore_mask(similarity, ignore_mask):
    if ignore_mask is not None and ignore_mask.any():
        similarity[ignore_mask] = -np.inf
    return similarity


class SimilarityComputer(ABC):
    @abstractmethod
    def compute(self, query, database, pair_selector, ignore_mask):
        raise NotImplementedError


class SinglePipelineComputer(SimilarityComputer):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def compute(self, query, database, pair_selector, ignore_mask):
        similarity = self.pipeline(query, database)
        return _apply_ignore_mask(similarity, ignore_mask)


class WildfusionComputer(SimilarityComputer):
    def __init__(self, wildfusion_similarity):
        self.wildfusion_similarity = wildfusion_similarity

    def compute(self, query, database, pair_selector, ignore_mask):
        self.wildfusion_similarity.set_pair_selector(pair_selector)
        return self.wildfusion_similarity(query, database)
