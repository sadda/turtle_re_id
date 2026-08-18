from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from wildlife_datasets.datasets import WildlifeDataset


@dataclass
class Split:
    name: str
    mask_database: pd.Series
    mask_query: pd.Series

    def database(self, wildlife_dataset: WildlifeDataset) -> WildlifeDataset:
        # TODO: return the same object if index full
        return wildlife_dataset.get_subset(self.mask_database)

    def query(self, wildlife_dataset: WildlifeDataset) -> WildlifeDataset:
        return wildlife_dataset.get_subset(self.mask_query)


class Splitter(ABC):
    @abstractmethod
    def get_split(self, metadata: pd.DataFrame) -> Split:
        raise NotImplementedError

    @abstractmethod
    def get_split_name(self) -> str:
        """Name of the split this splitter produces, without needing any metadata."""
        raise NotImplementedError


class YearSplitter(Splitter):
    def __init__(self, split_year: int, year_col: str = "year", identity_col: str = "identity") -> None:
        self.split_year = split_year
        self.year_col = year_col
        self.identity_col = identity_col

    def get_split(self, metadata: pd.DataFrame) -> Split:
        # TODO: this removes new individuals. Should not be here
        mask_database = metadata[self.year_col] < self.split_year
        labels_database = metadata.loc[mask_database, self.identity_col].to_numpy()
        mask_query = metadata[self.year_col] == self.split_year
        mask_query = mask_query & metadata[self.identity_col].isin(np.unique(labels_database))
        return Split(name=self.get_split_name(), mask_database=mask_database, mask_query=mask_query)

    def get_split_name(self) -> str:
        return str(self.split_year)


class DiscoverySplitter(Splitter):
    def get_split(self, metadata: pd.DataFrame) -> Split:
        mask = pd.Series(True, index=metadata.index)
        return Split(name=self.get_split_name(), mask_database=mask, mask_query=mask)

    def get_split_name(self) -> str:
        return "discovery"
