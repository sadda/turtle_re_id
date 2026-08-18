from wildlife_datasets.datasets import WildlifeDataset

from ..datasets.datasets import MatchingDataset
from .cache import SimilarityCache
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
        splitter: Splitter,
        run_specs: list[SimilarityRunSpec],
        cache: SimilarityCache,
        cache_prefix: str | None = None,
        device: str = "cuda",
    ) -> None:
        self.matching_dataset = matching_dataset
        self.wildlife_dataset = wildlife_dataset
        self.splitter = splitter
        self.run_specs = run_specs
        self.cache = cache
        self.cache_prefix = cache_prefix
        self.device = device

    def run(self) -> None:
        splits_by_name = {split.name: split for split in self.splitter.get_splits(self.wildlife_dataset.metadata)}

        # TODO: this is a bit strange because it is significantly different from SimilarityJob
        built_pipeline_specs: set[int] = set()
        for run_spec in self.run_specs:
            pipeline_spec = run_spec.pipeline_spec
            # TODO: Potentially build and discard one by one, to save GPU memory
            if id(pipeline_spec) not in built_pipeline_specs:
                pipeline_spec.build(self.device, self.cache_prefix)
                built_pipeline_specs.add(id(pipeline_spec))

            job = SimilarityJob(
                matching_dataset=self.matching_dataset,
                wildlife_dataset=self.wildlife_dataset,
                split=splits_by_name[run_spec.split_name],
                run_spec=run_spec,
                cache=self.cache,
            )
            job.run()
