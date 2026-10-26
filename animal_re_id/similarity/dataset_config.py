from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..models import (
    get_pipeline_aliked,
    get_pipeline_dino,
    get_pipeline_disk,
    get_pipeline_miew,
)
from .ignore_pairs_policy import AllowAllPairsPolicy, CombinedIgnorePairsPolicy, IgnorePairsPolicy, MetadataIgnorePairsPolicy
from .pipeline_spec import PipelineSpec, SinglePipelineSpec, WildfusionPipelineSpec
from .run_spec import SimilarityRunSpec
from .splitters import ClosedSetYearSplitter, DiscoverySplitter, MultiSeedClosedSetSplitter, Splitter


@dataclass
class SimilarityDatasetConfig:
    matching_parts: Sequence[str]
    pipeline_specs: Sequence[PipelineSpec]
    ignore_pairs_policies: Sequence[IgnorePairsPolicy]
    splitters: Sequence[Splitter]
    img_load: str = "bbox"

    def run_specs(self) -> list[SimilarityRunSpec]:
        return [
            SimilarityRunSpec(pipeline_spec, ignore_policy, splitter, matching_part, self.img_load)
            for pipeline_spec in self.pipeline_specs
            for ignore_policy in self.ignore_pairs_policies
            for splitter in self.splitters
            for matching_part in self.matching_parts
        ]


def get_similarity_dataset_config(dataset_name: str, B: int = 100, img_load: str = "bbox") -> SimilarityDatasetConfig:
    """
    Config shared by similarity compute and analysis: enough to build cache keys
    (pipeline names/B, ignore-policy cache keys, split names) without instantiating
    any model or touching a dataset. Only compute needs pipeline_spec.build().
    """
    if dataset_name == "TurtlewatchEgypt":
        return SimilarityDatasetConfig(
            matching_parts=["bodies", "heads", "front flippers", "hind flippers"],
            pipeline_specs=[
                SinglePipelineSpec("miew", get_pipeline_miew, B=B),
                WildfusionPipelineSpec("miew-aliked", get_pipeline_miew, get_pipeline_aliked, B=B),
            ],
            ignore_pairs_policies=[
                MetadataIgnorePairsPolicy(cache_key=False),
                AllowAllPairsPolicy(cache_key=True),
            ],
            splitters=[ClosedSetYearSplitter(int(split_year)) for split_year in np.arange(2018, 2026)],
            img_load=img_load,
        )
    elif dataset_name == "SeaTurtleID2022":
        return SimilarityDatasetConfig(
            matching_parts=["bodies", "heads", "front flippers", "hind flippers"],
            pipeline_specs=[
                SinglePipelineSpec("dino", get_pipeline_dino, B=B),
                WildfusionPipelineSpec("dino-aliked", get_pipeline_dino, get_pipeline_aliked, B=B),
            ],
            ignore_pairs_policies=[
                MetadataIgnorePairsPolicy(cache_key=False),
                AllowAllPairsPolicy(cache_key=True),
            ],
            splitters=[ClosedSetYearSplitter(int(split_year)) for split_year in np.arange(2011, 2023)],
            img_load=img_load,
        )
    elif dataset_name == "TurtlesOfSMSRC":
        return SimilarityDatasetConfig(
            matching_parts=["heads", "front flippers", "hind flippers"],
            pipeline_specs=[
                WildfusionPipelineSpec("miew-aliked", get_pipeline_miew, get_pipeline_aliked, B=B),
                WildfusionPipelineSpec("miew-disk", get_pipeline_miew, get_pipeline_disk, B=B),
            ],
            ignore_pairs_policies=[CombinedIgnorePairsPolicy(cache_key=None)],
            splitters=[DiscoverySplitter()],
            img_load=img_load,
        )
    elif dataset_name == "StripeSpotter":
        return SimilarityDatasetConfig(
            matching_parts=["bodies", "flanks"],
            pipeline_specs=[
                SinglePipelineSpec("dino", get_pipeline_dino, B=B),
                WildfusionPipelineSpec("dino-aliked", get_pipeline_dino, get_pipeline_aliked, B=B),
            ],
            ignore_pairs_policies=[AllowAllPairsPolicy(cache_key=None)],
            splitters=MultiSeedClosedSetSplitter(0.5, seed=666, n_splits=5).build(),
            img_load=img_load,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
