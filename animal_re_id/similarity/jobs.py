from wildlife_datasets.datasets import WildlifeDataset

from ..datasets.datasets import MatchingDataset
from .cache import SimilarityCache
from .ignore_pairs_policy import IgnorePairsPolicy
from .pipeline_spec import PipelineSpec
from .result import SimilarityResult
from .run_spec import SimilarityRunSpec
from .splitters import Split, Splitter


class SimilarityJob:
    def __init__(
        self,
        matching_dataset: MatchingDataset,
        wildlife_dataset: WildlifeDataset,
        split: Split,
        run_spec: SimilarityRunSpec,
        cache: SimilarityCache,
    ) -> None:
        self.matching_dataset = matching_dataset
        self.wildlife_dataset = wildlife_dataset
        self.split = split
        self.run_spec = run_spec
        self.cache = cache

    def run(self) -> None:
        key_parts = self.run_spec.key(self.cache)
        if self.cache.exists(key_parts):
            return

        database = self.split.database(self.wildlife_dataset)
        query = self.split.query(self.wildlife_dataset)

        pipeline_spec = self.run_spec.pipeline_spec
        ignore_policy = self.run_spec.ignore_policy
        matching_part = self.run_spec.matching_part

        pair_selector = ignore_policy.get_pair_selector(self.matching_dataset, matching_part)
        similarity = pipeline_spec.compute(query, database, pair_selector)

        result = SimilarityResult(
            similarity=similarity,
            query=query,
            database=database,
            name=pipeline_spec.name,
            B=pipeline_spec.B,
            img_load=self.run_spec.img_load,
            matching_part=matching_part,
            match_opposite_sides=ignore_policy.cache_key,
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
        img_load = self.wildlife_dataset.img_load

        for pipeline_spec in self.pipeline_specs:
            # TODO: this is a bit strange because it is significantly different from SimilarityJob
            pipeline_spec.build(self.device, self.cache_prefix)

            for ignore_policy in self.ignore_pairs_policies:
                for split in splits:
                    run_spec = SimilarityRunSpec(pipeline_spec, ignore_policy, split.name, self.matching_part, img_load)
                    job = SimilarityJob(
                        matching_dataset=self.matching_dataset,
                        wildlife_dataset=self.wildlife_dataset,
                        split=split,
                        run_spec=run_spec,
                        cache=self.cache,
                    )
                    job.run()
