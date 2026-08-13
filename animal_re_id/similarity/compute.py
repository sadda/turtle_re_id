from abc import ABC, abstractmethod

import numpy as np


def _mask_ignore_pairs(similarity, ignore_pairs):
    if ignore_pairs:
        ignore_pairs_arr = np.asarray(ignore_pairs)
        similarity[ignore_pairs_arr[:, 0], ignore_pairs_arr[:, 1]] = -np.inf
    return similarity


class SimilarityComputer(ABC):
    @abstractmethod
    def compute(self, query, database, ignore_pairs):
        raise NotImplementedError


class SinglePipelineComputer(SimilarityComputer):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def compute(self, query, database, ignore_pairs):
        similarity = self.pipeline(query, database)
        return _mask_ignore_pairs(similarity, ignore_pairs)


class WildfusionComputer(SimilarityComputer):
    def __init__(self, wildfusion_similarity):
        self.wildfusion_similarity = wildfusion_similarity

    def compute(self, query, database, ignore_pairs):
        self.wildfusion_similarity.set_ignore_pairs(ignore_pairs)
        similarity = self.wildfusion_similarity(query, database)
        return _mask_ignore_pairs(similarity, ignore_pairs)
