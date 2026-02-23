import pandas as pd
from wildlife_datasets.datasets import TurtlewatchEgypt_New

def merge_datasets(
        master_root,
        roots,
        load_segmentation=True,
        encounter_prefixes=None,
        **kwargs
        ):

    dfs = []
    if encounter_prefixes is None:
        encounter_prefixes = range(len(roots))
    for root, encounter_prefix in zip(roots, encounter_prefixes):
        dataset = TurtlewatchEgypt_New(f'{master_root}/{root}', load_segmentation=load_segmentation)
        dataset.df['path'] = root.split('/')[-1] + '/' + dataset.df['path']
        dataset.df['encounter_id'] = dataset.df['encounter_id'].apply(lambda x: f'{encounter_prefix}_{x}')
        dfs.append(dataset.df)
    df = pd.concat(dfs).reset_index(drop=True)
    # TODO: image_id should be unique
    df['image_id'] = range(len(df))
    return TurtlewatchEgypt_New(root=master_root, df=df, **kwargs)