from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from animal_re_id.similarity.ignore_pairs_policy import IgnorePairsPolicy
from animal_re_id.similarity.pipeline_spec import PipelineSpec
from animal_re_id.similarity.run_spec import SimilarityRunSpec
from animal_re_id.similarity.splitters import Splitter


def make_pipeline_spec(name: str = "dino", B: int | None = 5) -> PipelineSpec:
    return cast(PipelineSpec, SimpleNamespace(name=name, B=B))


def make_ignore_policy(cache_key: object = "allow_all") -> IgnorePairsPolicy:
    return cast(IgnorePairsPolicy, SimpleNamespace(cache_key=cache_key))


def make_splitter(split_name: str = "2021") -> Splitter:
    return cast(Splitter, SimpleNamespace(get_split_name=lambda: split_name))


class TestKeyParts:
    def test_key_parts_order_and_values(self) -> None:
        run_spec = SimilarityRunSpec(make_pipeline_spec(), make_ignore_policy(), make_splitter(), "heads", "bbox")
        assert run_spec.key_parts() == ["dino", 5, "bbox", "heads", "allow_all", "2021"]

    def test_key_parts_with_none_B(self) -> None:
        pipeline_spec = make_pipeline_spec(name="aliked", B=None)
        run_spec = SimilarityRunSpec(pipeline_spec, make_ignore_policy(), make_splitter(), "bodies", "full")
        assert run_spec.key_parts() == ["aliked", None, "full", "bodies", "allow_all", "2021"]


class TestLoad:
    def test_load_delegates_to_cache_load_with_key_parts(self) -> None:
        run_spec = SimilarityRunSpec(make_pipeline_spec(), make_ignore_policy(), make_splitter(), "heads", "bbox")
        cache = MagicMock(name="cache")
        cache.load.return_value = {"similarity": "matrix"}

        result = run_spec.load(cache)

        cache.load.assert_called_once_with(run_spec.key_parts())
        assert result == {"similarity": "matrix"}
