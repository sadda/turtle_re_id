import os
import pickle
import warnings

import numpy as np

from animal_re_id.datasets import TurtlewatchEgypt_New, get_ignore_pairs, get_subset
from animal_re_id.models import Wildfusion_Similarity, get_pipeline_aliked, get_pipeline_mega, get_pipeline_miew

warnings.filterwarnings("ignore")


file_name = "../notebooks/individuals.csv"
root = "/data/wildlife_datasets/turtle_watch/citizen"
root_heads = "/data/wildlife_datasets/turtle_watch/PROFILES"
root_flippers_f = "/data/wildlife_datasets/turtle_watch/TWE_flipperprofiles_registered ind_LR_front"
root_flippers_r = "/data/wildlife_datasets/turtle_watch/TWE_flipperprofiles_registered ind_LR_rear"
root_new = "/data/wildlife_datasets/turtle_watch"
roots_new = [
    "2017 PICS + DATASHEET",
    "2018 PICS + DATASHEET",
    "2019 PICS + DATASHEET",
    "2020 PICS + DATASHEET",
    "2021 PICS + DATASHEET",
    "2022 PICS + online database",
    "2023 PICS + online database",
    "2024 PICS + online database",
    "2025 PICS + online database",
]
root_features = "/data/wildlife_datasets/features"
root_results = "/data/wildlife_datasets/twe_similarity/data_selection"

pipeline_funs = [
    (get_pipeline_mega, None, "mega"),
    (get_pipeline_miew, None, "miew"),
    (get_pipeline_miew, get_pipeline_aliked, "miew-aliked"),
]

matching_parts = ["heads", "front flippers", "rear flippers"]
img_loads = ["bbox", "bbox_mask"]
dataset_name = "twe"
split_years = np.arange(2018, 2026).tolist()
Bs = [5, 10, 50, 100]
device = "cuda"


for B in Bs:
    predictions_all = {}
    scores_all = {}
    for img_load in img_loads:
        database_query_matching = True
        dataset_kwargs = {"img_load": img_load}
        dataset = TurtlewatchEgypt_New(root_new, roots_new, database_query_matching, file_name, dataset_kwargs=dataset_kwargs)

        for matching_part in matching_parts:
            dataset_subset = get_subset(dataset, matching_part)
            dataset_subset.metadata["year"] = dataset_subset.metadata["path"].str[:4].astype(int)
            mask = dataset_subset.metadata["identity"] != "unknown"
            dataset_subset = dataset_subset.get_subset(mask)

            cache_prefix = f"{root_features}/{dataset_name}_{matching_part}_{img_load}"
            for pipeline1_fun, pipeline2_fun, name in pipeline_funs:
                if pipeline2_fun is None:
                    pipeline = pipeline1_fun(device=device, cache_prefix=cache_prefix)
                else:
                    pipeline1 = pipeline1_fun(device=device, cache_prefix=cache_prefix)
                    pipeline2 = pipeline2_fun(device=device, cache_prefix=cache_prefix)
                    pipeline = Wildfusion_Similarity(pipeline1, pipeline2, B)

                for match_opposite_sides in [False, True]:
                    for split_year in split_years:
                        # Skip if already computed
                        file_name_save = f"{name}_{B}_{img_load}_{matching_part}_{match_opposite_sides}_{split_year}"
                        if os.path.exists(file_name_save):
                            continue

                        # Extract the database
                        mask_database = dataset_subset.metadata["year"] < split_year
                        database = dataset_subset.get_subset(mask_database)
                        
                        # Extract the query set
                        labels_database = database.metadata["identity"].to_numpy()
                        mask_query1 = dataset_subset.metadata["year"] == split_year
                        mask_query2 = np.isin(dataset_subset.metadata["identity"], np.unique(labels_database))
                        query = dataset_subset.get_subset(mask_query1 & mask_query2)

                        # Extract ignore_pairs
                        if match_opposite_sides:
                            ignore_pairs = []
                        else:
                            cols_equal = dataset.matching_parts[matching_part]["cols_equal"]
                            cols_unequal = dataset.matching_parts[matching_part]["cols_unequal"]
                            ignore_pairs = get_ignore_pairs(query.metadata, database.metadata, cols_equal, cols_unequal)

                        if pipeline2_fun is not None:
                            assert isinstance(pipeline, Wildfusion_Similarity)
                            pipeline.set_ignore_pairs(ignore_pairs)

                        # Compute the similarity matrix
                        similarity = pipeline(query, database)

                        # Fix the similarity matrix by ignore_pairs
                        if ignore_pairs:
                            ignore_pairs_arr = np.asarray(ignore_pairs)
                            similarity[ignore_pairs_arr[:, 0], ignore_pairs_arr[:, 1]] = -np.inf

                        # Save the data
                        save_data = {
                            "similarity": similarity,
                            "query": query,
                            "database": database,
                            "B": B,
                            "img_load": img_load,
                            "matching_part": matching_part,
                            "name": name,
                            "match_opposite_sides": match_opposite_sides,
                            "split_year": split_year,
                        }
                        with open(f"{root_results}/{file_name_save}.pkl", "wb") as file:
                            pickle.dump(save_data, file)
