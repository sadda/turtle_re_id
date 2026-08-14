import os
import warnings

import numpy as np

from animal_re_id.config import load_paths, resolve_path
from animal_re_id.datasets import SeaTurtleID2022, TurtlewatchEgypt_New, get_wildlife_dataset
from animal_re_id.models import get_pipeline_aliked, get_pipeline_dino, get_pipeline_miew
from animal_re_id.similarity import (
    AllowAllPairsPolicy,
    MetadataIgnorePairsPolicy,
    SimilarityCache,
    SimilarityGrid,
    SinglePipelineSpec,
    WildfusionPipelineSpec,
    YearSplitter,
)

warnings.filterwarnings("ignore")


dataset_name = "TurtlewatchEgypt"
# dataset_name = "SeaTurtleID2022"

matching_parts = ["bodies", "heads", "front flippers", "rear flippers"]
img_load = "bbox"
B = 100
device = "cuda"

paths = load_paths(dataset_name, local_path="config/paths.json")

wildlife_datasets = {}
database_query_matching = True

if dataset_name == "TurtlewatchEgypt":
    split_years = np.arange(2018, 2026).tolist()
    matching_dataset_class = TurtlewatchEgypt_New
    pipeline_specs = [
        SinglePipelineSpec("miew", get_pipeline_miew, B=B),
        WildfusionPipelineSpec("miew-aliked", get_pipeline_miew, get_pipeline_aliked, B=B),
    ]

    def postprocess(wildlife_dataset):
        wildlife_dataset.metadata["year"] = wildlife_dataset.metadata["path"].str[:4].astype(int)
        mask = wildlife_dataset.metadata["identity"] != "unknown"
        return wildlife_dataset.get_subset(mask)
elif dataset_name == "SeaTurtleID2022":
    split_years = np.arange(2011, 2023).tolist()
    matching_dataset_class = SeaTurtleID2022
    pipeline_specs = [
        SinglePipelineSpec("dino", get_pipeline_dino, B=B),
        WildfusionPipelineSpec("dino-aliked", get_pipeline_dino, get_pipeline_aliked, B=B),
    ]

    def postprocess(wildlife_dataset):
        return wildlife_dataset
else:
    raise ValueError("Dataset now known")

dataset_kwargs = {"img_load": img_load}
matching_dataset = matching_dataset_class.from_paths(paths, database_query_matching, dataset_kwargs=dataset_kwargs)
for matching_part in matching_parts:
    wildlife_dataset = get_wildlife_dataset(matching_dataset, matching_part)
    wildlife_datasets[matching_part] = postprocess(wildlife_dataset)

root_features = resolve_path(paths["root_features"])
root_similarity = resolve_path(paths["root_similarity"])
os.makedirs(root_similarity, exist_ok=True)

splitter = YearSplitter(split_years)
cache = SimilarityCache(root_similarity)
ignore_pairs_policies = [
    MetadataIgnorePairsPolicy(cache_key=False),
    AllowAllPairsPolicy(cache_key=True),
]

for matching_part in matching_parts:
    wildlife_dataset = wildlife_datasets[matching_part]
    cache_prefix = f"{root_features}/{matching_part}_{img_load}"

    grid = SimilarityGrid(
        matching_dataset=matching_dataset,
        wildlife_dataset=wildlife_dataset,
        matching_part=matching_part,
        splitter=splitter,
        pipeline_specs=pipeline_specs,
        ignore_pairs_policies=ignore_pairs_policies,
        cache=cache,
        device=device,
        cache_prefix=cache_prefix,
    )
    grid.run()
