from types import SimpleNamespace
from typing import cast

import pandas as pd
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.similarity import MaskedPairSelector, TopkPairSelector

from animal_re_id.datasets.datasets import MatchingDataset
from animal_re_id.similarity.ignore_pairs_policy import (
    AllowAllPairsPolicy,
    MetadataIgnorePairsPolicy,
)


def make_matching_dataset(cols_equal: list[str], cols_unequal: list[str], matching_part: str = "heads") -> MatchingDataset:
    return cast(
        MatchingDataset,
        SimpleNamespace(matching_parts={matching_part: {"cols_equal": cols_equal, "cols_unequal": cols_unequal}}),
    )


def make_wildlife_dataset_like(metadata: pd.DataFrame) -> WildlifeDataset:
    return cast(WildlifeDataset, SimpleNamespace(metadata=metadata))


class TestAllowAllPairsPolicy:
    def test_cache_key_is_stored(self) -> None:
        policy = AllowAllPairsPolicy(cache_key="allow_all")
        assert policy.cache_key == "allow_all"

    def test_get_pair_selector_returns_topk(self) -> None:
        policy = AllowAllPairsPolicy(cache_key="allow_all")
        selector = policy.get_pair_selector(matching_dataset=cast(MatchingDataset, None), matching_part="heads")
        assert isinstance(selector, TopkPairSelector)

    def test_get_ignore_mask_is_none(self) -> None:
        policy = AllowAllPairsPolicy(cache_key="allow_all")
        mask = policy.get_ignore_mask(
            matching_dataset=cast(MatchingDataset, None),
            matching_part="heads",
            query=cast(WildlifeDataset, None),
            database=cast(WildlifeDataset, None),
        )
        assert mask is None


class TestMetadataIgnorePairsPolicy:
    def test_get_pair_selector_returns_masked_selector(self) -> None:
        matching_dataset = make_matching_dataset(cols_equal=["identity"], cols_unequal=[])
        policy = MetadataIgnorePairsPolicy(cache_key="match_opposite")
        selector = policy.get_pair_selector(matching_dataset, "heads")
        assert isinstance(selector, MaskedPairSelector)

    def test_get_ignore_mask_ignores_same_identity(self) -> None:
        matching_dataset = make_matching_dataset(cols_equal=["identity"], cols_unequal=[])
        policy = MetadataIgnorePairsPolicy(cache_key="match_opposite")

        query = make_wildlife_dataset_like(pd.DataFrame({"identity": ["a", "b"]}))
        database = make_wildlife_dataset_like(pd.DataFrame({"identity": ["a", "c"]}))

        mask = policy.get_ignore_mask(matching_dataset, "heads", query, database)
        assert mask.shape == (2, 2)
        # query[0]="a" matches database[0]="a" -> ignored
        assert mask[0, 0]
        assert not mask[0, 1]
        assert not mask[1, 0]
        assert not mask[1, 1]

    def test_get_ignore_mask_ignores_different_values_in_cols_unequal(self) -> None:
        matching_dataset = make_matching_dataset(cols_equal=[], cols_unequal=["orientation"], matching_part="front flippers")
        policy = MetadataIgnorePairsPolicy(cache_key="match_same")

        query = make_wildlife_dataset_like(pd.DataFrame({"orientation": ["left", "right"]}))
        database = make_wildlife_dataset_like(pd.DataFrame({"orientation": ["left", "left"]}))

        mask = policy.get_ignore_mask(matching_dataset, "front flippers", query, database)
        # query[0]="left" vs database[0]="left" -> equal -> not ignored (cols_unequal ignores unequal pairs)
        assert not mask[0, 0]
        # query[0]="left" vs database[1]="left" -> equal -> not ignored
        assert not mask[0, 1]
        # query[1]="right" vs database[0]="left" -> unequal -> ignored
        assert mask[1, 0]
        assert mask[1, 1]

    def test_get_ignore_mask_uses_correct_matching_part(self) -> None:
        matching_dataset = cast(
            MatchingDataset,
            SimpleNamespace(
                matching_parts={
                    "heads": {"cols_equal": ["identity"], "cols_unequal": []},
                    "bodies": {"cols_equal": [], "cols_unequal": []},
                }
            ),
        )
        policy = MetadataIgnorePairsPolicy(cache_key="k")
        query = make_wildlife_dataset_like(pd.DataFrame({"identity": ["a"]}))
        database = make_wildlife_dataset_like(pd.DataFrame({"identity": ["a"]}))

        mask_heads = policy.get_ignore_mask(matching_dataset, "heads", query, database)
        mask_bodies = policy.get_ignore_mask(matching_dataset, "bodies", query, database)
        assert mask_heads[0, 0]
        assert not mask_bodies[0, 0]

    def test_cache_key_is_stored(self) -> None:
        policy = MetadataIgnorePairsPolicy(cache_key="match_opposite")
        assert policy.cache_key == "match_opposite"
