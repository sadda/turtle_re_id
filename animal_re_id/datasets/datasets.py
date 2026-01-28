import numpy as np
import pandas as pd
from typing import List
from wildlife_datasets import datasets as wd_datasets


class Dataset:
    def __init__(
            self,
            matching_parts: List[str],
            is_database: bool,
            ignore_unknown: bool = True
            ):
        
        self.matching_parts = matching_parts
        self.is_database = is_database
        self.ignore_unknown = ignore_unknown
    
    def _get_ignore_pairs(self, df, cols_equal, cols_unequal):
        ignore_pairs = []

        # Ignores same values in cols_equal
        for col in cols_equal:
            s = df[col]
            if self.ignore_unknown and col == 'identity':
                s = s[s != 'unknown']
            
            groups = s.groupby(s).groups
            for idx in groups.values():
                idx = idx.to_numpy()
                i, j = np.meshgrid(idx, idx, indexing='ij')
                ignore_pairs.extend(zip(i.ravel(), j.ravel()))

        # Ignores different values in cols_unequal
        for col in cols_unequal:
            s = df[col]
            if self.ignore_unknown and col == 'identity':
                s = s[s != 'unknown']

            values = s.to_numpy()
            idx = s.index.to_numpy()

            mask = values[:, None] != values[None, :]
            i, j = np.where(mask)
            ignore_pairs.extend(zip(idx[i], idx[j]))

        return ignore_pairs

    def get_datasets_subset(self, col, min_value, max_value):
        for matching_part, dataset in self.datasets.items():
            s = dataset.metadata[col]
            mask = (s >= min_value) & (s <= max_value)            
            self.datasets[matching_part] = dataset.get_subset(mask)

    def load_datasets(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclasses.")

    def load_ignore_pairs(self):
        if self.is_database:
            self.ignore_pairs = {matching_part: None for matching_part in self.datasets}
        else:
            self.ignore_pairs = {}
            for matching_part, dataset in self.datasets.items():
                cols_equal = self.matching_parts[matching_part]['cols_equal']
                cols_unequal = self.matching_parts[matching_part]['cols_unequal']
                self.ignore_pairs[matching_part] = self._get_ignore_pairs(dataset.metadata, cols_equal, cols_unequal)

    def set_unknown_identity(self):
        for dataset in self.datasets.values():
            dataset.metadata["identity_orig"] = dataset.metadata["identity"].copy()
            dataset.metadata["identity"] = "unknown"


class Dataset_Ultralytics(Dataset):
    def __init__(self, matching_parts, is_database, score_min=0.8, **kwargs):
        self.score_min = score_min
        super().__init__(matching_parts, is_database, **kwargs)

    def load_datasets(self):
        self.datasets = {}
        for matching_part in self.matching_parts:
            labels = self.matching_parts[matching_part]['labels']
            idx1 = self.dataset_full.metadata['score'] >= self.score_min
            idx2 = self.dataset_full.metadata['label'].isin(labels)
            self.datasets[matching_part] = self.dataset_full.get_subset(idx1 & idx2)


class TurtlesOfSMSRC(Dataset_Ultralytics):
    def __init__(self, root, is_database, **kwargs):
        matching_parts = {
            'heads': {'labels': ['head'], 'cols_equal': ['identity'], 'cols_unequal': []},
            'front flippers': {'labels': ['flipper_fl', 'flipper_fr'], 'cols_equal': ['identity'], 'cols_unequal': ['label']},
            'rear flippers': {'labels': ['flipper_rl', 'flipper_rr'], 'cols_equal': ['identity'], 'cols_unequal': ['label']}
        }
        self.dataset_full = wd_datasets.TurtlesOfSMSRC(root, img_load='bbox', load_segmentation=True, load_label=True)
        super().__init__(matching_parts, is_database, **kwargs)


class SeaTurtleID2022(Dataset):
    def __init__(self, root, is_database, **kwargs):
        matching_parts = {
            'heads': {'cols_equal': ['identity'], 'cols_unequal': []},
            'front flippers': {'cols_equal': ['identity'], 'cols_unequal': ['orientation']},
            'rear flippers': {'cols_equal': ['identity'], 'cols_unequal': ['orientation']}
        }
        self.dataset_heads = wd_datasets.SeaTurtleID2022(root, category_name='head', img_load='bbox', load_label=True)
        self.dataset_flippers = wd_datasets.SeaTurtleID2022(root, category_name='flipper', img_load='bbox', load_label=True)
        self.dataset_flippers.metadata['orientation'] = self.dataset_flippers.metadata['orientation'].replace({
            'front_left': 'flipper_fl',
            'front_right': 'flipper_fr',
            'rear_left': 'flipper_rl',
            'rear_right': 'flipper_rr',
        })
        self.add_year(self.dataset_heads)
        self.add_year(self.dataset_flippers)
        super().__init__(matching_parts, is_database, **kwargs)

    def add_year(self, dataset):
        dates = pd.to_datetime(dataset.metadata['date'])
        dataset.metadata['year'] = dates.apply(lambda x: x.year)
    
    def load_datasets(self):
        idx_front = self.dataset_flippers.metadata['orientation'].isin(['flipper_fl', 'flipper_fr'])
        idx_rear = self.dataset_flippers.metadata['orientation'].isin(['flipper_rl', 'flipper_rr'])

        self.datasets = {}
        self.datasets['heads'] = self.dataset_heads
        self.datasets['front flippers'] = self.dataset_flippers.get_subset(idx_front)
        self.datasets['rear flippers'] = self.dataset_flippers.get_subset(idx_rear)
