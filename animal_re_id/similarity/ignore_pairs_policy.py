from abc import ABC, abstractmethod

from wildlife_tools.similarity import MaskedPairSelector, MetadataIgnoreMask, PairSelector, TopkPairSelector


class IgnorePairsPolicy(ABC):
    def __init__(self, cache_key):
        self.cache_key = cache_key

    @abstractmethod
    def get_pair_selector(self, dataset, matching_part) -> PairSelector:
        raise NotImplementedError

    @abstractmethod
    def get_ignore_mask(self, dataset, matching_part, query, database):
        raise NotImplementedError


class NoIgnorePairsPolicy(IgnorePairsPolicy):
    def get_pair_selector(self, dataset, matching_part):
        return TopkPairSelector()

    def get_ignore_mask(self, dataset, matching_part, query, database):
        return None


class ColumnIgnorePairsPolicy(IgnorePairsPolicy):
    def get_pair_selector(self, dataset, matching_part):
        return MaskedPairSelector(TopkPairSelector(), self._build_mask_provider(dataset, matching_part))

    def get_ignore_mask(self, dataset, matching_part, query, database):
        mask_provider = self._build_mask_provider(dataset, matching_part)
        return mask_provider.get_ignore_mask(query.metadata, database.metadata)

    def _build_mask_provider(self, dataset, matching_part):
        cols_equal = dataset.matching_parts[matching_part]["cols_equal"]
        cols_unequal = dataset.matching_parts[matching_part]["cols_unequal"]
        return MetadataIgnoreMask(cols_equal, cols_unequal)
