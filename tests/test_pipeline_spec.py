from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import animal_re_id.similarity.pipeline_spec as pipeline_spec_module
from animal_re_id.similarity.pipeline_spec import (
    SinglePipelineSpec,
    WildfusionPipelineSpec,
    _apply_ignore_mask,
)


class FakeSimilarityPipeline:
    def __init__(self, similarity: NDArray[np.floating]) -> None:
        self.similarity = similarity
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, query: Any, database: Any) -> NDArray[np.floating]:
        self.calls.append((query, database))
        return self.similarity.copy()


class TestApplyIgnoreMask:
    def test_no_mask_returns_similarity_unchanged(self) -> None:
        similarity = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _apply_ignore_mask(similarity, None)
        assert np.array_equal(result, similarity)

    def test_mask_all_false_is_noop(self) -> None:
        similarity = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.zeros((2, 2), dtype=bool)
        result = _apply_ignore_mask(similarity, mask)
        assert np.array_equal(result, similarity)

    def test_masked_entries_set_to_negative_infinity(self) -> None:
        similarity = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [False, True]])
        result = _apply_ignore_mask(similarity, mask)
        assert result[0, 0] == -np.inf
        assert result[1, 1] == -np.inf
        assert result[0, 1] == 2.0
        assert result[1, 0] == 3.0


class TestSinglePipelineSpec:
    def test_compute_before_build_raises(self) -> None:
        spec = SinglePipelineSpec("dino", pipeline_fun=lambda **kw: FakeSimilarityPipeline(np.zeros((1, 1))))
        with pytest.raises(RuntimeError, match="dino"):
            spec.compute("q", "d", pair_selector=None, ignore_mask=None)

    def test_build_invokes_pipeline_fun_with_device_and_cache_prefix(self) -> None:
        received: dict[str, Any] = {}

        def pipeline_fun(device: str, cache_prefix: str | None) -> FakeSimilarityPipeline:
            received["device"] = device
            received["cache_prefix"] = cache_prefix
            return FakeSimilarityPipeline(np.zeros((1, 1)))

        spec = SinglePipelineSpec("dino", pipeline_fun=pipeline_fun)
        spec.build(device="cuda", cache_prefix="prefix")
        assert received == {"device": "cuda", "cache_prefix": "prefix"}

    def test_compute_returns_pipeline_output_with_mask_applied(self) -> None:
        similarity = np.array([[1.0, 2.0], [3.0, 4.0]])
        spec = SinglePipelineSpec("dino", pipeline_fun=lambda **kw: FakeSimilarityPipeline(similarity))
        spec.build(device="cpu", cache_prefix=None)
        mask = np.array([[True, False], [False, False]])
        result = spec.compute("query", "database", pair_selector=None, ignore_mask=mask)
        assert result[0, 0] == -np.inf
        assert result[0, 1] == 2.0

    def test_compute_passes_query_and_database_through(self) -> None:
        pipeline = FakeSimilarityPipeline(np.array([[1.0]]))
        spec = SinglePipelineSpec("dino", pipeline_fun=lambda **kw: pipeline)
        spec.build(device="cpu", cache_prefix=None)
        spec.compute("the_query", "the_database", pair_selector=None, ignore_mask=None)
        assert pipeline.calls == [("the_query", "the_database")]

    def test_name_and_B_are_stored(self) -> None:
        spec = SinglePipelineSpec("dino", pipeline_fun=lambda **kw: None, B=7)
        assert spec.name == "dino"
        assert spec.B == 7


class FakeWildfusionSimilarity:
    instances: list["FakeWildfusionSimilarity"] = []

    def __init__(self, pipeline1: Any, pipeline2: Any, B: int | None) -> None:
        self.pipeline1 = pipeline1
        self.pipeline2 = pipeline2
        self.B = B
        self.pair_selector: Any = None
        self.call_args: tuple[Any, Any] | None = None
        FakeWildfusionSimilarity.instances.append(self)

    def set_pair_selector(self, pair_selector: Any) -> None:
        self.pair_selector = pair_selector

    def __call__(self, query: Any, database: Any) -> NDArray[np.floating]:
        self.call_args = (query, database)
        return np.array([[0.5, -0.5], [0.1, 0.9]])


class TestWildfusionPipelineSpec:
    @pytest.fixture(autouse=True)
    def patch_wildfusion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeWildfusionSimilarity.instances = []
        monkeypatch.setattr(pipeline_spec_module, "Wildfusion_Similarity", FakeWildfusionSimilarity)

    def test_compute_before_build_raises(self) -> None:
        spec = WildfusionPipelineSpec("fusion", pipeline1_fun=lambda **kw: "p1", pipeline2_fun=lambda **kw: "p2")
        with pytest.raises(RuntimeError, match="fusion"):
            spec.compute("q", "d", pair_selector=None, ignore_mask=None)

    def test_build_constructs_both_pipelines_and_wildfusion(self) -> None:
        calls: list[tuple[str, str, str | None]] = []

        def pipeline1_fun(device: str, cache_prefix: str | None) -> str:
            calls.append(("p1", device, cache_prefix))
            return "pipeline1"

        def pipeline2_fun(device: str, cache_prefix: str | None) -> str:
            calls.append(("p2", device, cache_prefix))
            return "pipeline2"

        spec = WildfusionPipelineSpec("fusion", pipeline1_fun, pipeline2_fun, B=3)
        spec.build(device="cuda", cache_prefix="pre")

        assert calls == [("p1", "cuda", "pre"), ("p2", "cuda", "pre")]
        fusion = FakeWildfusionSimilarity.instances[0]
        assert fusion.pipeline1 == "pipeline1"
        assert fusion.pipeline2 == "pipeline2"
        assert fusion.B == 3

    def test_compute_sets_pair_selector_and_applies_mask(self) -> None:
        spec = WildfusionPipelineSpec("fusion", lambda **kw: "p1", lambda **kw: "p2")
        spec.build(device="cpu", cache_prefix=None)
        pair_selector = object()
        mask = np.array([[True, False], [False, False]])

        result = spec.compute("query", "database", pair_selector=pair_selector, ignore_mask=mask)

        fusion = FakeWildfusionSimilarity.instances[0]
        assert fusion.pair_selector is pair_selector
        assert fusion.call_args == ("query", "database")
        assert result[0, 0] == -np.inf
        assert result[0, 1] == -0.5
