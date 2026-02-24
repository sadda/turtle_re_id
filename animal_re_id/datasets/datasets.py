import os
import pandas as pd
from wildlife_datasets import datasets as wd_datasets
from ..turtlewatch_egypt import merge_datasets


class Dataset:
    def __init__(self, dataset, matching_parts):
        self.dataset = dataset
        self.matching_parts = matching_parts
    

class TurtlesSegmented(Dataset):
    def __init__(self, dataset, database_query_matching):
        if database_query_matching:
            matching_parts = {
                'heads': {
                    'labels': ['head'],
                    'cols_equal': [], 
                    'cols_unequal': []
                    },
                'front flippers': {
                    'labels': ['flipper_fl', 'flipper_fr'],
                    'cols_equal': [],
                    'cols_unequal': []
                    },
                'rear flippers': {
                    'labels': ['flipper_rl', 'flipper_rr'],
                    'cols_equal': [],
                    'cols_unequal': []
                    }
                }
        else:
            matching_parts = {
                'heads': {
                    'labels': ['head'],
                    'cols_equal': ['identity', 'encounter_id'], 
                    'cols_unequal': []
                    },
                'front flippers': {
                    'labels': ['flipper_fl', 'flipper_fr'],
                    'cols_equal': ['identity', 'encounter_id'],
                    'cols_unequal': ['label', 'orientation']
                    },
                'rear flippers': {
                    'labels': ['flipper_rl', 'flipper_rr'],
                    'cols_equal': ['identity', 'encounter_id'],
                    'cols_unequal': ['label', 'orientation']
                    }
                }
        super().__init__(dataset, matching_parts)


class TurtlesOfSMSRC(TurtlesSegmented):
    def __init__(self, root, is_database, score_min=0.8):
        dataset = wd_datasets.TurtlesOfSMSRC(root, img_load='bbox', load_segmentation=True, load_label=True)
        mask = dataset.metadata['score'] >= score_min
        super().__init__(dataset.get_subset(mask), is_database)


class TurtlewatchEgypt_Citizen(TurtlesSegmented):
    def __init__(self, root, is_database, score_min=0.8):
        dataset = wd_datasets.TurtlewatchEgypt_Citizen(root, img_load='bbox', load_segmentation=True, load_label=True)
        mask = dataset.metadata['score'] >= score_min
        super().__init__(dataset.get_subset(mask), is_database)


class TurtlewatchEgypt_New(TurtlesSegmented):
    def __init__(self, root_master, roots, is_database, score_min=0.8):
        dataset = merge_datasets(root_master, roots, img_load='bbox', load_segmentation=True, load_label=True)
        mask = dataset.metadata['score'] >= score_min
        super().__init__(dataset.get_subset(mask), is_database)


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


class TurtlewatchEgypt_Master(Dataset):
    def __init__(self, root_heads, root_flippers_f, root_flippers_r, **kwargs):
        # TODO: does not support query-query matching
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
        
        metadata_h = wd_datasets.TurtlewatchEgypt_Master(root_heads, **kwargs).metadata
        metadata_f = wd_datasets.TurtlewatchEgypt_Master(root_flippers_f, **kwargs).metadata
        metadata_r = wd_datasets.TurtlewatchEgypt_Master(root_flippers_r, **kwargs).metadata

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
