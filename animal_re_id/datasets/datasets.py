import numpy as np
import os
import pandas as pd
from wildlife_datasets import datasets as wd_datasets
from ..turtlewatch_egypt import merge_datasets


class Dataset:
    def __init__(self, dataset, matching_parts):
        self.dataset = dataset
        self.matching_parts = matching_parts
    

class TurtlesOfSMSRC(Dataset):
    def __init__(self, root, score_min=0.8):
        matching_parts = {
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
        dataset = wd_datasets.TurtlesOfSMSRC(root, img_load='bbox', load_segmentation=True, load_label=True)
        mask = dataset.metadata['score'] >= score_min
        super().__init__(dataset.get_subset(mask), matching_parts)




# TODO: odsud vsechno prepsat

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


class TurtlewatchEgypt_New(Dataset):
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


class TurtlewatchEgypt_Master(Dataset):
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
        
        metadata_h = wd_datasets.TurtlewatchEgypt_Master(root_heads).metadata
        metadata_f = wd_datasets.TurtlewatchEgypt_Master(root_flippers_f).metadata
        metadata_r = wd_datasets.TurtlewatchEgypt_Master(root_flippers_r).metadata

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
        self.dataset = wd_datasets.TurtlewatchEgypt_Master(root=None, df=metadata, load_label=True)
