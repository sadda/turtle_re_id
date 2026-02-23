import numpy as np
import os
import pandas as pd
from wildlife_datasets import datasets as wd_datasets
from wildlife_datasets.datasets import TurtlewatchEgypt_Master
from ..turtlewatch_egypt import merge_datasets

def get_datasets_subset(dataset, col, min_value, max_value):
    s = dataset.metadata[col]
    mask = (s >= min_value) & (s <= max_value)            
    return dataset.get_subset(mask)

def get_ignore_pairs(df, cols_equal, cols_unequal, ignore_unknown=True):
    ignore_pairs = []

    # Ignores same values in cols_equal
    for col in cols_equal:
        s = df[col]
        if ignore_unknown and col == 'identity':
            s = s[s != 'unknown']
        
        groups = s.groupby(s).groups
        for idx in groups.values():
            idx = idx.to_numpy()
            i, j = np.meshgrid(idx, idx, indexing='ij')
            ignore_pairs.extend(zip(i.ravel(), j.ravel()))

    # Ignores different values in cols_unequal
    for col in cols_unequal:
        s = df[col]
        if ignore_unknown and col == 'identity':
            s = s[s != 'unknown']

        values = s.to_numpy()
        idx = s.index.to_numpy()

        mask = values[:, None] != values[None, :]
        i, j = np.where(mask)
        ignore_pairs.extend(zip(idx[i], idx[j]))

    return ignore_pairs

def set_unknown_identity(dataset):
    dataset.metadata["identity_orig"] = dataset.metadata["identity"].copy()
    dataset.metadata["identity"] = "unknown"


class Dataset:
    def get_ignore_pairs(self, matching_part, dataset=None, **kwargs):
        if dataset is None:
            dataset = self.get_subset(matching_part)
        cols_equal = self.matching_parts[matching_part]['cols_equal']
        cols_unequal = self.matching_parts[matching_part]['cols_unequal']
        return get_ignore_pairs(dataset.metadata, cols_equal, cols_unequal, **kwargs)

    def get_subset(self, matching_part):
        labels = self.matching_parts[matching_part]['labels']
        mask = self.dataset.metadata['label'].isin(labels)
        return self.dataset.get_subset(mask)

    def get_subsets(self, matching_parts):
        return {m: self.get_subset(m) for m in matching_parts}

    def keep_only_unknown(self):
        mask = self.dataset.metadata["identity"] == "unknown"
        self.dataset = self.dataset.get_subset(mask)

    def remove_unknown(self):
        mask = self.dataset.metadata["identity"] != "unknown"
        self.dataset = self.dataset.get_subset(mask)


class TurtlesOfSMSRC(Dataset):
    def __init__(self, root, score_min=0.8):
        self.matching_parts = {
            'heads': {
                'labels': ['head'],
                'cols_equal': ['identity'], 
                'cols_unequal': []
                },
            'front flippers': {
                'labels': ['flipper_fl', 'flipper_fr'],
                'cols_equal': ['identity'],
                'cols_unequal': ['label']
                },
            'rear flippers': {
                'labels': ['flipper_rl', 'flipper_rr'],
                'cols_equal': ['identity'],
                'cols_unequal': ['label']
                }
            }
        self.dataset = wd_datasets.TurtlesOfSMSRC(root, img_load='bbox', load_segmentation=True, load_label=True)
        mask = self.dataset.metadata['score'] >= score_min
        self.dataset = self.dataset.get_subset(mask)


class SeaTurtleID2022(Dataset):
    def __init__(self, root):
        self.matching_parts = {
            'heads': {
                'labels': ['head'],
                'cols_equal': ['identity', 'encounter_id'],
                'cols_unequal': []
                },
            'front flippers': {
                'labels': ['front flipper'],
                'cols_equal': ['identity', 'encounter_id'],
                'cols_unequal': ['orientation']
                },
            'rear flippers': {
                'labels': ['rear flipper'],
                'cols_equal': ['identity', 'encounter_id'],
                'cols_unequal': ['orientation']
                }
            }
        
        metadata_h = wd_datasets.SeaTurtleID2022(root, category_name='head').metadata
        metadata_f = wd_datasets.SeaTurtleID2022(root, category_name='flipper').metadata
        
        metadata_h['image_id'] = 'h_' + metadata_h['image_id'].astype(str)
        metadata_h['label'] = 'head'
        metadata_f['image_id'] = 'f_' + metadata_f['image_id'].astype(str)
        mask = metadata_f['orientation'].astype(str).str.startswith('front')
        metadata_f.loc[mask, 'label'] = 'front flipper'
        mask = metadata_f['orientation'].astype(str).str.startswith('rear')
        metadata_f.loc[mask, 'label'] = 'rear flipper'

        metadata = pd.concat((metadata_h, metadata_f))
        dates = pd.to_datetime(metadata['date'])
        metadata['year'] = dates.apply(lambda x: x.year)
        metadata['encounter_id'] = metadata['identity'] + '_' + metadata['date']

        self.dataset = wd_datasets.SeaTurtleID2022(root, df=metadata, img_load='bbox', load_label=True)


class TurtleWatchEgypt(Dataset):
    def __init__(self, root_master, roots, score_min=0.8):
        self.matching_parts = {
            # TODO: add 'cols_equal': ['identity', 'encounter_id']?
            # TODO: change cols_unequal to orientation/label?
            'heads': {
                'labels': ['head'],
                'cols_equal': [],
                'cols_unequal': []
                },
            'front flippers': {
                'labels': ['flipper_fl', 'flipper_fr'],
                'cols_equal': ['date'],
                'cols_unequal': ['orientation']
                },
            'rear flippers': {
                'labels': ['flipper_rl', 'flipper_rr'],
                'cols_equal': ['date'],
                'cols_unequal': ['orientation']
                }
            }
        self.dataset = merge_datasets(root_master, roots, img_load='bbox', load_segmentation=True, load_label=True)
        mask = self.dataset.metadata['score'] >= score_min
        self.dataset = self.dataset.get_subset(mask)


class TurtleWatchEgypt_Master(Dataset):
    def __init__(self, root_heads, root_flippers_f, root_flippers_r):
        self.matching_parts = {
            'heads': {
                'labels': ['head'],
                'cols_equal': [],
                'cols_unequal': []
                },
            'front flippers': {
                'labels': ['front flipper'],
                'cols_equal': [],
                'cols_unequal': []
                },
            'rear flippers': {
                'labels': ['rear flipper'],
                'cols_equal': [],
                'cols_unequal': []
                }
            }
        
        metadata_h = TurtlewatchEgypt_Master(root_heads).metadata
        metadata_f = TurtlewatchEgypt_Master(root_flippers_f).metadata
        metadata_r = TurtlewatchEgypt_Master(root_flippers_r).metadata

        metadata_h['image_id'] = 'h_' + metadata_h['image_id'].astype(str)
        metadata_h['label'] = 'head'
        metadata_h['path'] = metadata_h['path'].apply(lambda x: os.path.join(root_heads, x))
        metadata_f['image_id'] = 'f_' + metadata_f['image_id'].astype(str)
        metadata_f['label'] = 'front flipper'
        metadata_f['path'] = metadata_f['path'].apply(lambda x: os.path.join(root_flippers_f, x))
        metadata_r['image_id'] = 'r_' + metadata_r['image_id'].astype(str)
        metadata_r['label'] = 'rear flipper'
        metadata_r['path'] = metadata_r['path'].apply(lambda x: os.path.join(root_flippers_r, x))
        
        metadata = pd.concat((metadata_h, metadata_f, metadata_r))
        self.dataset = TurtlewatchEgypt_Master(root=None, df=metadata, load_label=True)
