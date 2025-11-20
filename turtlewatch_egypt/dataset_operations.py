import pandas as pd
from .turtle_watch_egypt import TWE_NewData

def merge_datasets(master_root, roots, **kwargs):
    dfs = []
    n_encounters = 0
    for root in roots:
        dataset = TWE_NewData(f'{master_root}/{root}', load_segmentation=True)
        dataset.df['path'] = root.split('/')[-1] + '/' + dataset.df['path']
        dataset.df['encounter_id'] = dataset.df['encounter_id'] + n_encounters
        n_encounters += dataset.df['encounter_id'].nunique()
        dfs.append(dataset.df)
    df = pd.concat(dfs).reset_index(drop=True)
    df['image_id'] = range(len(df))
    return TWE_NewData(root=master_root, df=df, **kwargs)