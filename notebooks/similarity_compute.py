import os
import warnings

import numpy as np

from animal_re_id.config import load_paths, resolve_path
from animal_re_id.datasets import SeaTurtleID2022, TurtlewatchEgypt_New, get_subset
from animal_re_id.models import get_pipeline_aliked, get_pipeline_dino, get_pipeline_miew
from animal_re_id.similarity import (
    ColumnIgnorePairsPolicy,
    NoIgnorePairsPolicy,
    PipelineSpec,
    SimilarityCache,
    SimilarityGrid,
    YearSplitter,
)

warnings.filterwarnings("ignore")


dataset_name = "TurtlewatchEgypt"
# dataset_name = "SeaTurtleID2022"

matching_parts = ["bodies", "heads", "front flippers", "rear flippers"]
img_loads = ["bbox"]
B = 100
device = "cuda"

paths = load_paths(dataset_name, local_path="config/paths.json")

datasets = {}
datasets_subset = {}
database_query_matching = True

if dataset_name == "TurtlewatchEgypt":
    split_years = np.arange(2018, 2026).tolist()
    for img_load in img_loads:
        dataset_kwargs = {"img_load": img_load}
        dataset = TurtlewatchEgypt_New.from_paths(paths, database_query_matching, dataset_kwargs=dataset_kwargs)
        datasets[img_load] = dataset
        for matching_part in matching_parts:
            dataset_subset = get_subset(dataset, matching_part)
            dataset_subset.metadata["year"] = dataset_subset.metadata["path"].str[:4].astype(int)
            mask = dataset_subset.metadata["identity"] != "unknown"
            datasets_subset[(img_load, matching_part)] = dataset_subset.get_subset(mask)

    pipeline_specs = [
        PipelineSpec("miew", get_pipeline_miew, B=B),
        PipelineSpec("miew-aliked", get_pipeline_miew, get_pipeline_aliked, B=B),
    ]
elif dataset_name == "SeaTurtleID2022":
    split_years = np.arange(2011, 2023).tolist()
    for img_load in img_loads:
        dataset_kwargs = {"img_load": img_load}
        dataset = SeaTurtleID2022.from_paths(paths, database_query_matching, dataset_kwargs=dataset_kwargs)
        datasets[img_load] = dataset
        for matching_part in matching_parts:
            dataset_subset = get_subset(dataset, matching_part)
            datasets_subset[(img_load, matching_part)] = dataset_subset

    pipeline_specs = [
        PipelineSpec("dino", get_pipeline_dino, B=B),
        PipelineSpec("dino-aliked", get_pipeline_dino, get_pipeline_aliked, B=B),
    ]
else:
    raise ValueError("Dataset now known")

root_features = resolve_path(paths["root_features"])
root_similarity = resolve_path(paths["root_similarity"])
os.makedirs(root_similarity, exist_ok=True)

splitter = YearSplitter(split_years)
cache = SimilarityCache(root_similarity)
ignore_pairs_policies = [
    ColumnIgnorePairsPolicy(cache_key=False),
    NoIgnorePairsPolicy(cache_key=True),
]

for img_load in img_loads:
    dataset = datasets[img_load]
    for matching_part in matching_parts:
        dataset_subset = datasets_subset[(img_load, matching_part)]
        cache_prefix = f"{root_features}/{matching_part}_{img_load}"

        grid = SimilarityGrid(
            dataset=dataset,
            dataset_subset=dataset_subset,
            matching_part=matching_part,
            img_load=img_load,
            splitter=splitter,
            pipeline_specs=pipeline_specs,
            ignore_pairs_policies=ignore_pairs_policies,
            cache=cache,
            device=device,
            cache_prefix=cache_prefix,
        )
        grid.run()
