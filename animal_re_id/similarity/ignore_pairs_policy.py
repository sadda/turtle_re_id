from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from wildlife_tools.similarity import IgnoreMaskProvider, MaskedPairSelector, MetadataIgnoreMask, PairSelector, TopkPairSelector

from ..datasets.datasets import MatchingDataset


class IgnorePairsPolicy(ABC):
    def __init__(self, cache_key: Any) -> None:
        self.cache_key = cache_key

    @abstractmethod
    def get_pair_selector(self, matching_dataset: MatchingDataset, matching_part: str) -> PairSelector:
        raise NotImplementedError


class AllowAllPairsPolicy(IgnorePairsPolicy):
    def get_pair_selector(self, matching_dataset: MatchingDataset, matching_part: str) -> PairSelector:
        return TopkPairSelector()


class _DiagonalIgnoreMask(IgnoreMaskProvider):
    def get_ignore_mask(self, df1: pd.DataFrame, df2: pd.DataFrame) -> np.ndarray:
        return np.eye(len(df1), len(df2), dtype=bool)


class DiagonalIgnorePairsPolicy(IgnorePairsPolicy):
    def get_pair_selector(self, matching_dataset: MatchingDataset, matching_part: str) -> MaskedPairSelector:
        return MaskedPairSelector(TopkPairSelector(), _DiagonalIgnoreMask())


class MetadataIgnorePairsPolicy(IgnorePairsPolicy):
    def get_pair_selector(self, matching_dataset: MatchingDataset, matching_part: str) -> MaskedPairSelector:
        cols_equal = matching_dataset.matching_parts[matching_part]["cols_equal"]
        cols_unequal = matching_dataset.matching_parts[matching_part]["cols_unequal"]
        mask_provider = MetadataIgnoreMask(cols_equal, cols_unequal)
        return MaskedPairSelector(TopkPairSelector(), mask_provider)
