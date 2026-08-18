from dataclasses import dataclass
from typing import Any

from .cache import SimilarityCache
from .ignore_pairs_policy import IgnorePairsPolicy
from .pipeline_spec import PipelineSpec
from .splitters import Splitter


@dataclass
class SimilarityRunSpec:
    """Everything needed to identify and load one cached similarity result."""

    pipeline_spec: PipelineSpec
    ignore_policy: IgnorePairsPolicy
    splitter: Splitter
    matching_part: str
    img_load: str

    def key_parts(self) -> list[Any]:
        return [
            self.pipeline_spec.name,
            self.pipeline_spec.B,
            self.img_load,
            self.matching_part,
            self.ignore_policy.cache_key,
            self.splitter.get_split_name(),
        ]

    def load(self, cache: SimilarityCache) -> dict[str, Any]:
        return cache.load(self.key_parts())
