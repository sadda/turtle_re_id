from wildlife_datasets.datasets import WildlifeDataset

from ..datasets.datasets import Dataset
from .cache import SimilarityCache
from .compute import SimilarityComputer
from .ignore_pairs_policy import IgnorePairsPolicy
from .pipeline_spec import PipelineSpec
from .result import SimilarityResult
from .splitters import Split, Splitter


class SimilarityJob:
    def __init__(
        self,
        dataset: Dataset,
        dataset_subset: WildlifeDataset,
        matching_part: str,
        pipeline_spec: PipelineSpec,
        computer: SimilarityComputer,
        ignore_policy: IgnorePairsPolicy,
        split: Split,
        cache: SimilarityCache,
    ) -> None:
        self.dataset = dataset
        self.dataset_subset = dataset_subset
        self.matching_part = matching_part
        self.pipeline_spec = pipeline_spec
        self.computer = computer
        self.ignore_policy = ignore_policy
        self.split = split
        self.cache = cache

    def run(self) -> None:
        img_load = self.dataset_subset.img_load
        key_parts = self.cache.key_for(self.pipeline_spec, self.ignore_policy, self.split, img_load, self.matching_part)
        if self.cache.exists(key_parts):
            return

        database = self.split.database(self.dataset_subset)
        query = self.split.query(self.dataset_subset)

        pair_selector = self.ignore_policy.get_pair_selector(self.dataset, self.matching_part)
        ignore_mask = self.ignore_policy.get_ignore_mask(self.dataset, self.matching_part, query, database)
        similarity = self.computer.compute(query, database, pair_selector, ignore_mask)

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
        dataset: Dataset,
        dataset_subset: WildlifeDataset,
        matching_part: str,
        splitter: Splitter,
        pipeline_specs: list[PipelineSpec],
        ignore_pairs_policies: list[IgnorePairsPolicy],
        cache: SimilarityCache,
        device: str = "cuda",
        cache_prefix: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.dataset_subset = dataset_subset
        self.matching_part = matching_part
        self.splitter = splitter
        self.pipeline_specs = pipeline_specs
        self.ignore_pairs_policies = ignore_pairs_policies
        self.cache = cache
        self.device = device
        self.cache_prefix = cache_prefix

    def run(self) -> None:
        splits = self.splitter.get_splits(self.dataset_subset.metadata)

        for pipeline_spec in self.pipeline_specs:
            computer = pipeline_spec.build(self.device, self.cache_prefix)

            for ignore_policy in self.ignore_pairs_policies:
                for split in splits:
                    job = SimilarityJob(
                        dataset=self.dataset,
                        dataset_subset=self.dataset_subset,
                        matching_part=self.matching_part,
                        pipeline_spec=pipeline_spec,
                        computer=computer,
                        ignore_policy=ignore_policy,
                        split=split,
                        cache=self.cache,
                    )
                    job.run()
