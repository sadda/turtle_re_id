from collections.abc import Callable

import pandas as pd
from wildlife_datasets.datasets import WildlifeDataset

from animal_re_id.similarity.splitters import ClosedSetYearSplitter, OpenSetYearSplitter, Split


class TestSplit:
    def test_database_returns_subset_matching_mask(self, simple_wildlife_dataset: WildlifeDataset) -> None:
        mask_database = pd.Series([True, True, False, False, False, False])
        mask_query = pd.Series([False, False, False, False, True, True])
        split = Split(name="2021", mask_database=mask_database, mask_query=mask_query)

        database = split.database(simple_wildlife_dataset)
        assert list(database.metadata["identity"]) == ["a", "a"]

    def test_query_returns_subset_matching_mask(self, simple_wildlife_dataset: WildlifeDataset) -> None:
        mask_database = pd.Series([True, True, False, False, False, False])
        mask_query = pd.Series([False, False, False, False, True, True])
        split = Split(name="2021", mask_database=mask_database, mask_query=mask_query)

        query = split.query(simple_wildlife_dataset)
        assert list(query.metadata["identity"]) == ["c", "unknown"]

    def test_database_and_query_do_not_mutate_original(self, simple_wildlife_dataset: WildlifeDataset) -> None:
        original_len = len(simple_wildlife_dataset.metadata)
        mask = pd.Series([True] + [False] * 5)
        split = Split(name="x", mask_database=mask, mask_query=mask)
        split.database(simple_wildlife_dataset)
        assert len(simple_wildlife_dataset.metadata) == original_len


class TestClosedSetYearSplitter:
    def test_splits_by_year_boundary(self, simple_metadata: pd.DataFrame) -> None:
        splitter = ClosedSetYearSplitter(2021)
        split = splitter.get_split(simple_metadata)
        assert split.name == "2021-closed"
        # database: rows strictly before 2021 -> years 2019, 2020, 2019
        assert list(simple_metadata.loc[split.mask_database, "year"]) == [2019, 2020, 2019]

    def test_query_only_includes_identities_seen_in_database(self, simple_metadata: pd.DataFrame) -> None:
        splitter = ClosedSetYearSplitter(2021)
        split = splitter.get_split(simple_metadata)
        # query candidates are year == 2021 -> identities "b", "c", "unknown"
        # but only "b" appears in the database (years < 2021)
        assert list(simple_metadata.loc[split.mask_query, "identity"]) == ["b"]

    def test_different_splitter_instances_produce_independently_named_splits(self, simple_metadata: pd.DataFrame) -> None:
        splitter_2020 = ClosedSetYearSplitter(2020)
        splitter_2021 = ClosedSetYearSplitter(2021)
        assert splitter_2020.get_split(simple_metadata).name == "2020-closed"
        assert splitter_2021.get_split(simple_metadata).name == "2021-closed"

    def test_custom_year_and_identity_columns(self) -> None:
        metadata = pd.DataFrame(
            {
                "capture_year": [2020, 2020, 2021],
                "turtle_id": ["a", "b", "a"],
            }
        )
        splitter = ClosedSetYearSplitter(2021, year_col="capture_year", identity_col="turtle_id")
        split = splitter.get_split(metadata)
        assert list(metadata.loc[split.mask_database, "turtle_id"]) == ["a", "b"]
        assert list(metadata.loc[split.mask_query, "turtle_id"]) == ["a"]

    def test_no_new_identities_in_query_when_database_empty(self) -> None:
        metadata = pd.DataFrame({"year": [2021, 2021], "identity": ["a", "b"]})
        splitter = ClosedSetYearSplitter(2021)
        split = splitter.get_split(metadata)
        assert split.mask_query.sum() == 0

    def test_integration_with_split_dataset_methods(
        self,
        simple_metadata: pd.DataFrame,
        wildlife_dataset_factory: Callable[..., WildlifeDataset],
    ) -> None:
        wildlife_dataset = wildlife_dataset_factory(simple_metadata)
        splitter = ClosedSetYearSplitter(2021)
        split = splitter.get_split(simple_metadata)

        database = split.database(wildlife_dataset)
        query = split.query(wildlife_dataset)
        assert list(database.metadata["identity"]) == ["a", "a", "b"]
        assert list(query.metadata["identity"]) == ["b"]


class TestOpenSetYearSplitter:
    def test_splits_by_year_boundary(self, simple_metadata: pd.DataFrame) -> None:
        splitter = OpenSetYearSplitter(2021)
        split = splitter.get_split(simple_metadata)
        assert split.name == "2021-open"
        assert list(simple_metadata.loc[split.mask_database, "year"]) == [2019, 2020, 2019]

    def test_query_includes_identities_not_seen_in_database(self, simple_metadata: pd.DataFrame) -> None:
        splitter = OpenSetYearSplitter(2021)
        split = splitter.get_split(simple_metadata)
        # unlike ClosedSetYearSplitter, "c" and "unknown" (never seen before 2021) stay in the query
        assert list(simple_metadata.loc[split.mask_query, "identity"]) == ["b", "c", "unknown"]

    def test_query_kept_even_when_database_has_no_identities(self) -> None:
        metadata = pd.DataFrame({"year": [2021, 2021], "identity": ["a", "b"]})
        splitter = OpenSetYearSplitter(2021)
        split = splitter.get_split(metadata)
        assert split.mask_query.sum() == 2
