from types import SimpleNamespace
from typing import cast

import pandas as pd
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.similarity import MaskedPairSelector, TopkPairSelector

from animal_re_id.datasets.datasets import MatchingDataset
from animal_re_id.similarity.ignore_pairs_policy import (
    AllowAllPairsPolicy,
    DiagonalIgnorePairsPolicy,
    MetadataIgnorePairsPolicy,
)


def make_matching_dataset(cols_unequal: list[str], matching_part: str = "heads") -> MatchingDataset:
    return cast(
        MatchingDataset,
        SimpleNamespace(matching_parts={matching_part: {"cols_unequal": cols_unequal}}),
    )


def make_wildlife_dataset_like(metadata: pd.DataFrame) -> WildlifeDataset:
    return cast(WildlifeDataset, SimpleNamespace(metadata=metadata))


class TestAllowAllPairsPolicy:
    def test_cache_key_is_stored(self) -> None:
        policy = AllowAllPairsPolicy(cache_key="allow_all")
        assert policy.cache_key == "allow_all"

    def test_get_pair_selector_returns_topk(self) -> None:
        policy = AllowAllPairsPolicy(cache_key="allow_all")
        selector = policy.get_pair_selector(matching_dataset=cast(MatchingDataset, None), matching_part="heads", database_query_matching=True)
        assert isinstance(selector, TopkPairSelector)


class TestDiagonalIgnorePairsPolicy:
    def test_get_pair_selector_returns_masked_selector(self) -> None:
        policy = DiagonalIgnorePairsPolicy(cache_key="diagonal")
        selector = policy.get_pair_selector(matching_dataset=cast(MatchingDataset, None), matching_part="heads", database_query_matching=True)
        assert isinstance(selector, MaskedPairSelector)

    def test_get_ignore_mask_masks_diagonal(self) -> None:
        policy = DiagonalIgnorePairsPolicy(cache_key="diagonal")
        selector = policy.get_pair_selector(matching_dataset=cast(MatchingDataset, None), matching_part="heads", database_query_matching=True)

        query = make_wildlife_dataset_like(pd.DataFrame({"identity": ["a", "b"]}))
        database = make_wildlife_dataset_like(pd.DataFrame({"identity": ["a", "c"]}))

        mask = selector.get_ignore_mask(query, database)
        assert mask.tolist() == [[True, False], [False, True]]


class TestMetadataIgnorePairsPolicy:
    def test_get_pair_selector_returns_masked_selector(self) -> None:
        matching_dataset = make_matching_dataset(cols_unequal=[])
        policy = MetadataIgnorePairsPolicy(cache_key="match_opposite")
        selector = policy.get_pair_selector(matching_dataset, "heads", False)
        assert isinstance(selector, MaskedPairSelector)

    def test_get_ignore_mask_ignores_same_identity_for_query_query_matching(self) -> None:
        matching_dataset = make_matching_dataset(cols_unequal=[])
        policy = MetadataIgnorePairsPolicy(cache_key="match_opposite")
        selector = policy.get_pair_selector(matching_dataset, "heads", False)

        query = make_wildlife_dataset_like(pd.DataFrame({"identity": ["a", "b"]}))
        database = make_wildlife_dataset_like(pd.DataFrame({"identity": ["a", "c"]}))

        mask = selector.get_ignore_mask(query, database)
        assert mask.shape == (2, 2)
        # query[0]="a" matches database[0]="a" -> ignored
        assert mask[0, 0]
        assert not mask[0, 1]
        assert not mask[1, 0]
        assert not mask[1, 1]

    def test_get_ignore_mask_ignores_different_values_in_cols_unequal(self) -> None:
        matching_dataset = make_matching_dataset(cols_unequal=["orientation"], matching_part="front flippers")
        policy = MetadataIgnorePairsPolicy(cache_key="match_same")
        selector = policy.get_pair_selector(matching_dataset, "front flippers", True)

        query = make_wildlife_dataset_like(pd.DataFrame({"orientation": ["left", "right"]}))
        database = make_wildlife_dataset_like(pd.DataFrame({"orientation": ["left", "left"]}))

        mask = selector.get_ignore_mask(query, database)
        # query[0]="left" vs database[0]="left" -> equal -> not ignored (cols_unequal ignores unequal pairs)
        assert not mask[0, 0]
        # query[0]="left" vs database[1]="left" -> equal -> not ignored
        assert not mask[0, 1]
        # query[1]="right" vs database[0]="left" -> unequal -> ignored
        assert mask[1, 0]
        assert mask[1, 1]

    def test_get_ignore_mask_ignores_same_identity_only_for_query_query_matching(self) -> None:
        matching_dataset = make_matching_dataset(cols_unequal=[])
        policy = MetadataIgnorePairsPolicy(cache_key="k")
        query = make_wildlife_dataset_like(pd.DataFrame({"identity": ["a"]}))
        database = make_wildlife_dataset_like(pd.DataFrame({"identity": ["a"]}))

        mask_query_query = policy.get_pair_selector(matching_dataset, "heads", False).get_ignore_mask(query, database)
        mask_query_database = policy.get_pair_selector(matching_dataset, "heads", True).get_ignore_mask(query, database)
        assert mask_query_query[0, 0]
        assert not mask_query_database[0, 0]

    def test_cache_key_is_stored(self) -> None:
        policy = MetadataIgnorePairsPolicy(cache_key="match_opposite")
        assert policy.cache_key == "match_opposite"
