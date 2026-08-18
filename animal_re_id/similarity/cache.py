import os
import pickle
from typing import Any

from .ignore_pairs_policy import IgnorePairsPolicy
from .pipeline_spec import PipelineSpec


class SimilarityCache:
    def __init__(self, root: str) -> None:
        self.root = root

    def key_for(
        self,
        pipeline_spec: PipelineSpec,
        ignore_policy: IgnorePairsPolicy,
        split_name: str,
        img_load: str,
        matching_part: str,
    ) -> list[Any]:
        return [
            pipeline_spec.name,
            pipeline_spec.B,
            img_load,
            matching_part,
            ignore_policy.cache_key,
            split_name,
        ]

    def path(self, key_parts: list[Any]) -> str:
        file_name = "_".join(str(part) for part in key_parts)
        return os.path.join(self.root, f"{file_name}.pkl")

    def exists(self, key_parts: list[Any]) -> bool:
        return os.path.exists(self.path(key_parts))

    def save(self, key_parts: list[Any], data: dict[str, Any]) -> None:
        with open(self.path(key_parts), "wb") as file:
            pickle.dump(data, file)

    def load(self, key_parts: list[Any]) -> dict[str, Any]:
        with open(self.path(key_parts), "rb") as file:
            return pickle.load(file)

    def load_result(
        self,
        pipeline_spec: PipelineSpec,
        ignore_policy: IgnorePairsPolicy,
        split_name: str,
        img_load: str,
        matching_part: str,
    ) -> dict[str, Any]:
        return self.load(self.key_for(pipeline_spec, ignore_policy, split_name, img_load, matching_part))
