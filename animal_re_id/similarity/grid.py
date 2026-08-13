from .cache import SimilarityCache
from .ignore_pairs_policy import IgnorePairsPolicy
from .jobs import SimilarityJob
from .pipeline_spec import PipelineSpec
from .splitters import Splitter


class SimilarityGrid:
    def __init__(
        self,
        dataset,
        dataset_subset,
        matching_part,
        img_load,
        splitter: Splitter,
        pipeline_specs: list[PipelineSpec],
        ignore_pairs_policies: list[IgnorePairsPolicy],
        cache: SimilarityCache,
        device="cuda",
        cache_prefix=None,
    ):
        self.dataset = dataset
        self.dataset_subset = dataset_subset
        self.matching_part = matching_part
        self.img_load = img_load
        self.splitter = splitter
        self.pipeline_specs = pipeline_specs
        self.ignore_pairs_policies = ignore_pairs_policies
        self.cache = cache
        self.device = device
        self.cache_prefix = cache_prefix

    def run(self):
        splits = self.splitter.get_splits(self.dataset_subset.metadata)

        for pipeline_spec in self.pipeline_specs:
            computer = pipeline_spec.build(self.device, self.cache_prefix)

            for ignore_policy in self.ignore_pairs_policies:
                for split in splits:
                    job = SimilarityJob(
                        dataset=self.dataset,
                        dataset_subset=self.dataset_subset,
                        matching_part=self.matching_part,
                        img_load=self.img_load,
                        pipeline_spec=pipeline_spec,
                        computer=computer,
                        ignore_policy=ignore_policy,
                        split=split,
                        cache=self.cache,
                    )
                    job.run()
