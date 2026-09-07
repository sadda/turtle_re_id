import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from wildlife_datasets.datasets import WildlifeDataset

logger = logging.getLogger(__name__)


@dataclass
class Split:
    name: str
    mask_database: pd.Series
    mask_query: pd.Series

    def _get_subset(self, mask: pd.Series, wildlife_dataset: WildlifeDataset) -> WildlifeDataset:
        if len(mask) == len(wildlife_dataset) and mask.all():
            return wildlife_dataset
        else:
            return wildlife_dataset.get_subset(mask)

    def database(self, wildlife_dataset: WildlifeDataset) -> WildlifeDataset:
        return self._get_subset(self.mask_database, wildlife_dataset)

    def query(self, wildlife_dataset: WildlifeDataset) -> WildlifeDataset:
        return self._get_subset(self.mask_query, wildlife_dataset)


class Splitter(ABC):
    #: Whether query and database are disjoint sets for this splitter (True), or the
    #: same set (False, e.g. discovery-style self-matching).
    database_query_matching: bool

    @abstractmethod
    def get_split(self, metadata: pd.DataFrame) -> Split:
        raise NotImplementedError

    @abstractmethod
    def get_split_name(self) -> str:
        """Name of the split this splitter produces, without needing any metadata."""
        raise NotImplementedError


class YearSplitter(Splitter, ABC):
    database_query_matching = True

    def __init__(self, split_year: int, year_col: str = "year", identity_col: str = "identity") -> None:
        self.split_year = split_year
        self.year_col = year_col
        self.identity_col = identity_col

    def get_split(self, metadata: pd.DataFrame) -> Split:
        mask_database = metadata[self.year_col] < self.split_year
        mask_query = metadata[self.year_col] == self.split_year
        mask_query = self._filter_query(mask_query, mask_database, metadata)
        return Split(name=self.get_split_name(), mask_database=mask_database, mask_query=mask_query)

    @abstractmethod
    def _filter_query(self, mask_query: pd.Series, mask_database: pd.Series, metadata: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class ClosedSetYearSplitter(YearSplitter):
    """Query only includes identities already seen in the database (pure re-id accuracy)."""

    def _filter_query(self, mask_query: pd.Series, mask_database: pd.Series, metadata: pd.DataFrame) -> pd.Series:
        labels_database = metadata.loc[mask_database, self.identity_col].to_numpy()
        return mask_query & metadata[self.identity_col].isin(np.unique(labels_database))

    def get_split_name(self) -> str:
        return f"{self.split_year}-closed"


class OpenSetYearSplitter(YearSplitter):
    """Query may include identities never seen in the database (novelty/new-individual detection)."""

    def _filter_query(self, mask_query: pd.Series, mask_database: pd.Series, metadata: pd.DataFrame) -> pd.Series:
        return mask_query

    def get_split_name(self) -> str:
        return f"{self.split_year}-open"


class DisjointSetYearSplitter(YearSplitter):
    """Query only includes identities never seen in the database."""

    def _filter_query(self, mask_query: pd.Series, mask_database: pd.Series, metadata: pd.DataFrame) -> pd.Series:
        labels_database = metadata.loc[mask_database, self.identity_col].to_numpy()
        return mask_query & (~metadata[self.identity_col].isin(np.unique(labels_database)))

    def get_split_name(self) -> str:
        return f"{self.split_year}-disjoint"


class ColumnValueSplitter(Splitter):
    """Splits by a metadata column: rows matching query_value go to query, rows matching database_value go to database."""

    database_query_matching = True

    def __init__(self, column_name: str, query_value: Any, database_value: Any) -> None:
        self.column_name = column_name
        self.query_value = query_value
        self.database_value = database_value

    def get_split(self, metadata: pd.DataFrame) -> Split:
        column = metadata[self.column_name]
        other_values = set(column.unique()) - {self.query_value, self.database_value}
        if other_values or column.isnull().any():
            logger.warning(
                "Column %r has values outside {%r, %r}, excluded from the split: %s",
                self.column_name,
                self.query_value,
                self.database_value,
                sorted(other_values),
            )
        mask_query = column == self.query_value
        mask_database = column == self.database_value
        return Split(name=self.get_split_name(), mask_database=mask_database, mask_query=mask_query)

    def get_split_name(self) -> str:
        return f"{self.column_name}:{self.query_value}-vs-{self.database_value}"


class DiscoverySplitter(Splitter):
    database_query_matching = False

    def get_split(self, metadata: pd.DataFrame) -> Split:
        mask = pd.Series(True, index=metadata.index)
        return Split(name=self.get_split_name(), mask_database=mask, mask_query=mask)

    def get_split_name(self) -> str:
        return "discovery"
