from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.similarity import MaskedPairSelector, MetadataIgnoreMask, PairSelector, TopkPairSelector

from ..datasets.datasets import MatchingDataset


class IgnorePairsPolicy(ABC):
    def __init__(self, cache_key: Any) -> None:
        self.cache_key = cache_key

    @abstractmethod
    def get_pair_selector(self, matching_dataset: MatchingDataset, matching_part: str) -> PairSelector:
        raise NotImplementedError

    @abstractmethod
    def get_ignore_mask(
        self,
        matching_dataset: MatchingDataset,
        matching_part: str,
        query: WildlifeDataset,
        database: WildlifeDataset,
    ) -> np.ndarray | None:
        raise NotImplementedError


class AllowAllPairsPolicy(IgnorePairsPolicy):
    def get_pair_selector(self, matching_dataset: MatchingDataset, matching_part: str) -> PairSelector:
        return TopkPairSelector()

    def get_ignore_mask(
        self,
        matching_dataset: MatchingDataset,
        matching_part: str,
        query: WildlifeDataset,
        database: WildlifeDataset,
    ) -> np.ndarray | None:
        return None


class MetadataIgnorePairsPolicy(IgnorePairsPolicy):
    def get_pair_selector(self, matching_dataset: MatchingDataset, matching_part: str) -> PairSelector:
        return MaskedPairSelector(TopkPairSelector(), self._build_mask_provider(matching_dataset, matching_part))

    def get_ignore_mask(
        self,
        matching_dataset: MatchingDataset,
        matching_part: str,
        query: WildlifeDataset,
        database: WildlifeDataset,
    ) -> np.ndarray | None:
        mask_provider = self._build_mask_provider(matching_dataset, matching_part)
        return mask_provider.get_ignore_mask(query.metadata, database.metadata)

    def _build_mask_provider(self, matching_dataset: MatchingDataset, matching_part: str) -> MetadataIgnoreMask:
        cols_equal = matching_dataset.matching_parts[matching_part]["cols_equal"]
        cols_unequal = matching_dataset.matching_parts[matching_part]["cols_unequal"]
        return MetadataIgnoreMask(cols_equal, cols_unequal)
