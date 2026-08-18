from dataclasses import dataclass
from typing import Any

from .cache import SimilarityCache
from .ignore_pairs_policy import IgnorePairsPolicy
from .pipeline_spec import PipelineSpec


@dataclass
class SimilarityRunSpec:
    """Everything needed to identify and load one cached similarity result."""

    pipeline_spec: PipelineSpec
    ignore_policy: IgnorePairsPolicy
    split_name: str
    matching_part: str
    img_load: str

    def key(self, cache: SimilarityCache) -> list[Any]:
        return cache.key_for(self.pipeline_spec, self.ignore_policy, self.split_name, self.img_load, self.matching_part)

    def load(self, cache: SimilarityCache) -> dict[str, Any]:
        return cache.load_result(self.pipeline_spec, self.ignore_policy, self.split_name, self.img_load, self.matching_part)
