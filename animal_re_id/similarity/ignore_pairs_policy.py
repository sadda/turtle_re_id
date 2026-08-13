from abc import ABC, abstractmethod

from ..datasets import get_ignore_pairs


class IgnorePairsPolicy(ABC):
    def __init__(self, cache_key):
        self.cache_key = cache_key

    @abstractmethod
    def get_ignore_pairs(self, dataset, matching_part, query, database):
        raise NotImplementedError


class NoIgnorePairsPolicy(IgnorePairsPolicy):
    def get_ignore_pairs(self, dataset, matching_part, query, database):
        return []


class ColumnIgnorePairsPolicy(IgnorePairsPolicy):
    def get_ignore_pairs(self, dataset, matching_part, query, database):
        cols_equal = dataset.matching_parts[matching_part]["cols_equal"]
        cols_unequal = dataset.matching_parts[matching_part]["cols_unequal"]
        return get_ignore_pairs(query.metadata, database.metadata, cols_equal, cols_unequal)
