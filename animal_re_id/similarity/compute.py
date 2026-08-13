from abc import ABC, abstractmethod

import numpy as np
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.similarity import PairSelector, SimilarityPipeline

from ..models import Wildfusion_Similarity


def _apply_ignore_mask(similarity: np.ndarray, ignore_mask: np.ndarray | None) -> np.ndarray:
    if ignore_mask is not None and ignore_mask.any():
        similarity[ignore_mask] = -np.inf
    return similarity


class SimilarityComputer(ABC):
    @abstractmethod
    def compute(
        self,
        query: WildlifeDataset,
        database: WildlifeDataset,
        pair_selector: PairSelector,
        ignore_mask: np.ndarray | None,
    ) -> np.ndarray:
        raise NotImplementedError


class SinglePipelineComputer(SimilarityComputer):
    def __init__(self, pipeline: SimilarityPipeline) -> None:
        self.pipeline = pipeline

    def compute(
        self,
        query: WildlifeDataset,
        database: WildlifeDataset,
        pair_selector: PairSelector,
        ignore_mask: np.ndarray | None,
    ) -> np.ndarray:
        similarity = self.pipeline(query, database)
        return _apply_ignore_mask(similarity, ignore_mask)


class WildfusionComputer(SimilarityComputer):
    def __init__(self, wildfusion_similarity: Wildfusion_Similarity) -> None:
        self.wildfusion_similarity = wildfusion_similarity

    def compute(
        self,
        query: WildlifeDataset,
        database: WildlifeDataset,
        pair_selector: PairSelector,
        ignore_mask: np.ndarray | None,
    ) -> np.ndarray:
        self.wildfusion_similarity.set_pair_selector(pair_selector)
        return self.wildfusion_similarity(query, database)
