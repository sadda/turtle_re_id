import numpy as np
import pandas as pd
from wildlife_tools.similarity import CollectAll, MatchLightGlue
from wildlife_tools.similarity.pairwise.utils import visualise_matches

def extract_column(df, col):
    if col in df.columns:
        return df[col].to_numpy()
    else:
        return np.full(len(df), None)
    
def extract_features(dataset, pipeline, i):
    j = get_index(dataset, i)
    dataset = dataset.get_subset([j])
    dataset.transform = pipeline.transform.transforms[0]
    img, _ = dataset[0]
    dataset.transform = pipeline.transform
    features = pipeline.extractor(dataset)
    return img, features

def gather_df(
        dataset_q,
        dataset_d,
        i_sort,
        j_sort,
        similarity,
        name,
        matching_part
        ):
    
    image_ids_q = extract_column(dataset_q.df, 'image_id')
    identities_q = extract_column(dataset_q.df, 'identity')
    encounters_q = extract_column(dataset_q.df, 'encounter_id')
    image_ids_d = extract_column(dataset_d.df, 'image_id')
    identities_d = extract_column(dataset_d.df, 'identity')
    encounters_d = extract_column(dataset_d.df, 'encounter_id')
    result = pd.DataFrame({
        'encounter1': encounters_q[i_sort],
        'encounter2': encounters_d[j_sort],
        'identity1': identities_q[i_sort],
        'identity2': identities_d[j_sort],
        'image_id1': image_ids_q[i_sort],
        'image_id2': image_ids_d[j_sort],
        'matching_part': matching_part,
        'matcher': name,
        'similarity': similarity[i_sort, j_sort] / similarity.max(), 
        'similarity_orig': similarity[i_sort, j_sort],
    })
    return result

def get_index(dataset, image_id):
    idx = dataset.metadata['image_id'] == image_id
    if idx.sum() > 1:
        raise ValueError('image_id found multiple times.')
    elif idx.sum() == 1:
        return dataset.metadata[idx].index[0]
    elif idx.sum() == 0:
        print(f'image_id {image_id} not found')
        return None

def plot_matches(dataset_q, dataset_d, result):
    i_q = get_index(dataset_q, result['image_id1'])
    i_d = get_index(dataset_d, result['image_id2'])
    if dataset_q is dataset_d:
        identities = result[['identity1', 'identity2']].to_numpy()
        header_cols = [f'{identities[0]}_{identities[1]}', result['similarity']]
        dataset_d.plot_grid(idx=[i_q, i_d], n_cols=2, header_cols=header_cols, rotate=False)
    else:
        # TODO: this is hacky as fuck
        dataset_q = dataset_q.get_subset([i_q])
        if dataset_q.root:
            dataset_q.metadata['path'] = dataset_q.root + '/' + dataset_q.metadata['path']
        dataset_q.root = None

        dataset_d = dataset_d.get_subset([i_d])
        if dataset_d.root:
            dataset_d.metadata['path'] = dataset_d.root + '/' + dataset_d.metadata['path']
        dataset_d.root = None
        
        dataset_q.metadata = pd.concat((dataset_q.metadata, dataset_d.metadata))
        dataset_q.metadata = dataset_q.metadata.reset_index(drop=True)

        identities = result[['identity1', 'identity2']].to_numpy()
        header_cols = [f'{identities[0]}_{identities[1]}', result['similarity']]
        dataset_q.plot_grid(n_cols=2, header_cols=header_cols, rotate=False)

def visualise_match(
        dataset_q,
        dataset_d,
        result,
        pipeline,
        features_name,
        init_threshold=0.65,
        ax=None):

    img_q, features_q = extract_features(dataset_q, pipeline, result['image_id1'])
    img_d, features_d = extract_features(dataset_d, pipeline, result['image_id2'])

    matcher = MatchLightGlue(features=features_name, collector = CollectAll(), init_threshold=init_threshold)
    results = matcher(features_q, features_d)[0]
    visualise_matches(img_q, results['kpts0'], img_d, results['kpts1'], ax=ax)
