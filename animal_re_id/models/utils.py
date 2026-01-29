import pandas as pd
from wildlife_tools.similarity import CollectAll, MatchLightGlue
from wildlife_tools.similarity.pairwise.utils import visualise_matches

def gather_df(
        dataset,
        i_sort,
        j_sort,
        similarity,
        name,
        matching_part
        ):
    
    image_ids = dataset.df['image_id'].to_numpy()
    identities = dataset.df['identity'].to_numpy()
    encounters = dataset.df['encounter_id'].to_numpy()    
    result = pd.DataFrame({
        'encounter1': encounters[i_sort],
        'encounter2': encounters[j_sort],
        'identity1': identities[i_sort],
        'identity2': identities[j_sort],
        'image_id1': image_ids[i_sort],
        'image_id2': image_ids[j_sort],
        'index1': i_sort,
        'index2': j_sort,
        'matching_part': matching_part,
        'matcher': name,
        'similarity': similarity[i_sort, j_sort] / similarity.max(), 
        'similarity_orig': similarity[i_sort, j_sort],
    })
    return result

def plot_matches(dataset, result):
    idx = result[['index1', 'index2']].to_numpy().astype(int)
    identities = result[['identity1', 'identity2']].to_numpy()
    header_cols = [f'{identities[0]}_{identities[1]}', result['similarity']]
    dataset.plot_grid(idx=idx, n_cols=2, header_cols=header_cols, rotate=False)

def visualise_match(
        dataset,
        result,
        pipeline,
        features_name,
        init_threshold=0.65,
        ax=None):
    
    dataset_small = dataset.get_subset([result['index1'], result['index2']])
    dataset_small.transform = pipeline.transform.transforms[0]

    img0, _ = dataset_small[0]
    img1, _ = dataset_small[1]

    dataset_small.transform = pipeline.transform
    extractor = pipeline.extractor
    features = extractor(dataset_small)
    matcher = MatchLightGlue(features=features_name, collector = CollectAll(), init_threshold=init_threshold)
    results = matcher(features, features, pairs=[(0, 1)])[0]

    visualise_matches(img0, results['kpts0'], img1, results['kpts1'], ax=ax)