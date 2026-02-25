import os
import pandas as pd
from wildlife_datasets import datasets as wd_datasets
from ..turtlewatch_egypt import merge_datasets


def combine_matching_parts(matching_parts1, matching_parts2):
    assert matching_parts1.keys() == matching_parts2.keys()
    keys1 = matching_parts1.keys()
    keys2 = ['labels', 'cols_equal', 'cols_unequal']
    matching_parts = {key: {} for key in keys1}
    for key1 in keys1:
        for key2 in keys2:
            value1 = matching_parts1[key1][key2]
            value2 = matching_parts2[key1][key2]
            value = list(set(value1).union(set(value2)))
            matching_parts[key1][key2] = value
    return matching_parts


class Dataset:
    def __init__(self, dataset, matching_parts):
        self.dataset = dataset
        self.dataset.metadata = self.dataset.metadata.reset_index(drop = True)
        self.matching_parts = matching_parts
    
    def remove_unknown(self):
        mask = self.dataset.metadata["identity"] != "unknown"
        self.dataset = self.dataset.get_subset(mask)

    def keep_only_unknown(self):
        mask = self.dataset.metadata["identity"] == "unknown"
        self.dataset = self.dataset.get_subset(mask)


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


class TurtlesFlippersMerged(Dataset):
    def __init__(self, dataset, database_query_matching):
        if database_query_matching:
            matching_parts = {
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
        else:
            matching_parts = {
                'heads': {
                    'labels': ['head'],
                    'cols_equal': ['identity', 'encounter_id'],
                    'cols_unequal': []
                    },
                'front flippers': {
                    'labels': ['front flipper'],
                    'cols_equal': ['identity', 'encounter_id'],
                    'cols_unequal': ['label', 'orientation']
                    },
                'rear flippers': {
                    'labels': ['rear flipper'],
                    'cols_equal': ['identity', 'encounter_id'],
                    'cols_unequal': ['label', 'orientation']
                    }
                }
        super().__init__(dataset, matching_parts)


class TurtlesOfSMSRC(TurtlesSegmented):
    def __init__(self, root, is_database, score_min=0.8, **kwargs):
        dataset = wd_datasets.TurtlesOfSMSRC(root, img_load='bbox', load_segmentation=True, load_label=True, **kwargs)
        mask = dataset.metadata['score'] >= score_min
        super().__init__(dataset.get_subset(mask), is_database)


class TurtlewatchEgypt_Citizen(TurtlesSegmented):
    def __init__(self, root, is_database, score_min=0.8, **kwargs):
        dataset = wd_datasets.TurtlewatchEgypt_Citizen(root, img_load='bbox', load_segmentation=True, load_label=True, **kwargs)
        mask = dataset.metadata['score'] >= score_min
        super().__init__(dataset.get_subset(mask), is_database)


class TurtlewatchEgypt_New(TurtlesSegmented):
    def __init__(self, root_master, roots, is_database, file_name, score_min=0.8, **kwargs):
        dataset = merge_datasets(root_master, roots, file_name, img_load='bbox', load_segmentation=True, load_label=True, **kwargs)
        mask = dataset.metadata['score'] >= score_min
        super().__init__(dataset.get_subset(mask), is_database)


class SeaTurtleID2022(TurtlesFlippersMerged):
    def __init__(self, root, database_query_matching):
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

        dataset = wd_datasets.SeaTurtleID2022(root, df=metadata, img_load='bbox', load_label=True)
        super().__init__(dataset, database_query_matching)


class TurtlewatchEgypt_Master(TurtlesFlippersMerged):
    def __init__(self, root_heads, root_flippers_f, root_flippers_r, database_query_matching, file_name, score_min=0.8, **kwargs):
        metadata_h = wd_datasets.TurtlewatchEgypt_Master(root_heads, file_name=file_name, load_segmentation=True, **kwargs).metadata
        metadata_f = wd_datasets.TurtlewatchEgypt_Master(root_flippers_f, file_name=file_name, **kwargs).metadata
        metadata_r = wd_datasets.TurtlewatchEgypt_Master(root_flippers_r, file_name=file_name, **kwargs).metadata

        convertion = {
            'flipper_fr': 'front flipper',
            'flipper_fl': 'front flipper',
            'flipper_rl': 'rear flipper',
            'flipper_rr': 'rear flipper',
        }
        metadata_h['label'] = metadata_h['label'].replace(convertion)
        metadata_h['path'] = metadata_h['path'].apply(lambda x: os.path.join(root_heads, x))
        metadata_h = metadata_h[~metadata_h['label'].isnull()]
        metadata_h = metadata_h['score'] >= score_min
        metadata_f['label'] = 'front flipper'
        metadata_f['path'] = metadata_f['path'].apply(lambda x: os.path.join(root_flippers_f, x))
        metadata_r['label'] = 'rear flipper'
        metadata_r['path'] = metadata_r['path'].apply(lambda x: os.path.join(root_flippers_r, x))
        
        metadata = pd.concat((metadata_h, metadata_f, metadata_r))
        dataset = wd_datasets.TurtlewatchEgypt_Master(root=None, df=metadata, load_label=True, file_name=file_name, **kwargs)
        super().__init__(dataset, database_query_matching)


class TurtlewatchEgypt_Combined(Dataset):
    def __init__(
        self,
        root_master,
        roots,
        root_heads,
        root_flippers_f,
        root_flippers_r,
        file_name,
        score_min=0.8,
        **kwargs,
    ):

        dataset1 = TurtlewatchEgypt_New(root_master, roots, False, file_name, score_min=score_min, **kwargs)
        dataset2 = TurtlewatchEgypt_Master(root_heads, root_flippers_f, root_flippers_r, False, file_name, score_min=score_min, **kwargs)
        metadata = pd.concat((dataset1.dataset.metadata, dataset2.dataset.metadata))
        dataset = wd_datasets.WildlifeDataset(root=None, df=metadata, load_label=True, **kwargs)
        matching_parts = combine_matching_parts(dataset1.matching_parts, dataset2.matching_parts)
        super().__init__(dataset, matching_parts)
