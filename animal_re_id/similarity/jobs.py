from .cache import SimilarityCache
from .compute import SimilarityComputer
from .ignore_pairs_policy import IgnorePairsPolicy
from .pipeline_spec import PipelineSpec
from .result import SimilarityResult
from .splitters import Split


class SimilarityJob:
    def __init__(
        self,
        dataset,
        dataset_subset,
        matching_part,
        img_load,
        pipeline_spec: PipelineSpec,
        computer: SimilarityComputer,
        ignore_policy: IgnorePairsPolicy,
        split: Split,
        cache: SimilarityCache,
    ):
        self.dataset = dataset
        self.dataset_subset = dataset_subset
        self.matching_part = matching_part
        self.img_load = img_load
        self.pipeline_spec = pipeline_spec
        self.computer = computer
        self.ignore_policy = ignore_policy
        self.split = split
        self.cache = cache

    def run(self):
        key_parts = self.cache.key_for(self.pipeline_spec, self.ignore_policy, self.split, self.img_load, self.matching_part)
        if self.cache.exists(key_parts):
            return

        database = self.split.database(self.dataset_subset)
        query = self.split.query(self.dataset_subset)

        ignore_pairs = self.ignore_policy.get_ignore_pairs(self.dataset, self.matching_part, query, database)
        similarity = self.computer.compute(query, database, ignore_pairs)

        result = SimilarityResult(
            similarity=similarity,
            query=query,
            database=database,
            name=self.pipeline_spec.name,
            B=self.pipeline_spec.B,
            img_load=self.img_load,
            matching_part=self.matching_part,
            match_opposite_sides=self.ignore_policy.cache_key,
            split_year=self.split.name,
        )
        self.cache.save(key_parts, result.to_dict())
