from wildlife_datasets.datasets import WildlifeDataset

from ..datasets.datasets import MatchingDataset
from .cache import SimilarityCache
from .ignore_pairs_policy import IgnorePairsPolicy
from .pipeline_spec import PipelineSpec
from .result import SimilarityResult
from .splitters import Split, Splitter


class SimilarityJob:
    def __init__(
        self,
        matching_dataset: MatchingDataset,
        wildlife_dataset: WildlifeDataset,
        matching_part: str,
        pipeline_spec: PipelineSpec,
        ignore_policy: IgnorePairsPolicy,
        split: Split,
        cache: SimilarityCache,
    ) -> None:
        self.matching_dataset = matching_dataset
        self.wildlife_dataset = wildlife_dataset
        self.matching_part = matching_part
        self.pipeline_spec = pipeline_spec
        self.ignore_policy = ignore_policy
        self.split = split
        self.cache = cache

    def run(self) -> None:
        img_load = self.wildlife_dataset.img_load
        key_parts = self.cache.key_for(self.pipeline_spec, self.ignore_policy, self.split, img_load, self.matching_part)
        if self.cache.exists(key_parts):
            return

        database = self.split.database(self.wildlife_dataset)
        query = self.split.query(self.wildlife_dataset)

        pair_selector = self.ignore_policy.get_pair_selector(self.matching_dataset, self.matching_part)
        similarity = self.pipeline_spec.compute(query, database, pair_selector)

        result = SimilarityResult(
            similarity=similarity,
            query=query,
            database=database,
            name=self.pipeline_spec.name,
            B=self.pipeline_spec.B,
            img_load=img_load,
            matching_part=self.matching_part,
            match_opposite_sides=self.ignore_policy.cache_key,
            split_year=self.split.name,
        )
        self.cache.save(key_parts, result.to_dict())


class SimilarityGrid:
    def __init__(
        self,
        matching_dataset: MatchingDataset,
        wildlife_dataset: WildlifeDataset,
        matching_part: str,
        splitter: Splitter,
        pipeline_specs: list[PipelineSpec],
        ignore_pairs_policies: list[IgnorePairsPolicy],
        cache: SimilarityCache,
        device: str = "cuda",
        cache_prefix: str | None = None,
    ) -> None:
        self.matching_dataset = matching_dataset
        self.wildlife_dataset = wildlife_dataset
        self.matching_part = matching_part
        self.splitter = splitter
        self.pipeline_specs = pipeline_specs
        self.ignore_pairs_policies = ignore_pairs_policies
        self.cache = cache
        self.device = device
        self.cache_prefix = cache_prefix

    def run(self) -> None:
        splits = self.splitter.get_splits(self.wildlife_dataset.metadata)

        for pipeline_spec in self.pipeline_specs:
            # TODO: this is a bit strange because it is significantly different from SimilarityJob
            pipeline_spec.build(self.device, self.cache_prefix)

            for ignore_policy in self.ignore_pairs_policies:
                for split in splits:
                    job = SimilarityJob(
                        matching_dataset=self.matching_dataset,
                        wildlife_dataset=self.wildlife_dataset,
                        matching_part=self.matching_part,
                        pipeline_spec=pipeline_spec,
                        ignore_policy=ignore_policy,
                        split=split,
                        cache=self.cache,
                    )
                    job.run()
