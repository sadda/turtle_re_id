from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    name: str
    mask_database: pd.Series
    mask_query: pd.Series

    def database(self, dataset):
        return dataset.get_subset(self.mask_database)

    def query(self, dataset):
        return dataset.get_subset(self.mask_query)


class Splitter(ABC):
    @abstractmethod
    def get_splits(self, metadata: pd.DataFrame) -> list[Split]:
        raise NotImplementedError


class YearSplitter(Splitter):
    def __init__(self, split_years, year_col="year", identity_col="identity"):
        self.split_years = split_years
        self.year_col = year_col
        self.identity_col = identity_col

    def get_splits(self, metadata: pd.DataFrame) -> list[Split]:
        splits = []
        for split_year in self.split_years:
            # TODO: this removes new individuals. Should not be here
            mask_database = metadata[self.year_col] < split_year
            labels_database = metadata.loc[mask_database, self.identity_col].to_numpy()
            mask_query = metadata[self.year_col] == split_year
            mask_query = mask_query & metadata[self.identity_col].isin(np.unique(labels_database))
            splits.append(Split(name=str(split_year), mask_database=mask_database, mask_query=mask_query))
        return splits

