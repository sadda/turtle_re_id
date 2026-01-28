import pandas as pd

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
