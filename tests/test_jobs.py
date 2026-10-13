from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from animal_re_id.similarity.jobs import SimilarityGrid, SimilarityJob
from animal_re_id.similarity.run_spec import SimilarityRunSpec


def make_run_spec(matching_part: str = "heads") -> SimilarityRunSpec:
    pipeline_spec = MagicMock(name="pipeline_spec")
    pipeline_spec.name = "dino"
    pipeline_spec.B = 5
    pipeline_spec.compute.return_value = np.array([[1.0]])

    ignore_policy = MagicMock(name="ignore_policy")
    ignore_policy.cache_key = "allow_all"

    splitter = MagicMock(name="splitter")
    splitter.get_split_name.return_value = "2021"
    splitter.database_query_matching = True

    return SimilarityRunSpec(pipeline_spec, ignore_policy, splitter, matching_part, "bbox")


def make_job(cache_exists: bool = False) -> SimilarityJob:
    matching_dataset = MagicMock(name="matching_dataset")

    run_spec = make_run_spec()

    split = MagicMock(name="split")
    split.name = "2021"

    cache = MagicMock(name="cache")
    cache.exists.return_value = cache_exists

    job = SimilarityJob(
        matching_dataset=matching_dataset,
        split=split,
        run_spec=run_spec,
        cache=cache,
    )
    return job


class TestSimilarityJob:
    def test_run_skips_computation_when_cached(self) -> None:
        job = make_job(cache_exists=True)
        job.run()
        job.split.database.assert_not_called()
        job.split.query.assert_not_called()
        job.run_spec.pipeline_spec.compute.assert_not_called()
        job.cache.save.assert_not_called()

    def test_run_computes_and_saves_when_not_cached(self) -> None:
        job = make_job(cache_exists=False)
        job.split.database.return_value = "database_subset"
        job.split.query.return_value = "query_subset"
        job.run_spec.ignore_policy.get_pair_selector.return_value = "pair_selector"

        job.run()

        job.run_spec.pipeline_spec.compute.assert_called_once_with("query_subset", "database_subset", "pair_selector")
        job.cache.save.assert_called_once()
        saved_key, saved_data = job.cache.save.call_args[0]
        assert saved_key == job.run_spec.key_parts()
        assert saved_data["name"] == "dino"
        assert saved_data["B"] == 5
        assert saved_data["img_load"] == "bbox"
        assert saved_data["matching_part"] == "heads"
        assert saved_data["match_opposite_sides"] == "allow_all"
        assert saved_data["split_year"] == "2021"
        assert np.array_equal(saved_data["similarity"], np.array([[1.0]]))

    def test_run_uses_run_spec_key_parts_for_cache_key(self) -> None:
        job = make_job(cache_exists=True)
        job.run()
        job.cache.exists.assert_called_once_with(job.run_spec.key_parts())

    def test_run_builds_pair_selector_with_matching_dataset_and_part(self) -> None:
        job = make_job(cache_exists=False)
        job.split.database.return_value = "database_subset"
        job.split.query.return_value = "query_subset"

        job.run()

        job.run_spec.ignore_policy.get_pair_selector.assert_called_once_with(job.matching_dataset, "heads", True)

    def test_run_uses_matching_dataset_wildlife_dataset_for_split(self) -> None:
        job = make_job(cache_exists=False)
        job.run()
        job.split.database.assert_called_once_with(job.matching_dataset.wildlife_dataset)
        job.split.query.assert_called_once_with(job.matching_dataset.wildlife_dataset)


class TestSimilarityGrid:
    def _make_run_specs(
        self,
        n_specs: int,
        n_policies: int,
        n_splitters: int,
        matching_part: str = "heads",
    ) -> tuple[list[SimilarityRunSpec], list[MagicMock], list[MagicMock], list[MagicMock]]:
        pipeline_specs = [MagicMock(name=f"spec{i}") for i in range(n_specs)]
        ignore_policies = [MagicMock(name=f"policy{i}") for i in range(n_policies)]
        splitters = [MagicMock(name=f"splitter{i}") for i in range(n_splitters)]
        for i, splitter in enumerate(splitters):
            splitter.get_split.return_value = MagicMock(name=f"split{i}")

        run_specs = [SimilarityRunSpec(pipeline_spec, ignore_policy, splitter, matching_part, "bbox") for pipeline_spec in pipeline_specs for ignore_policy in ignore_policies for splitter in splitters]
        return run_specs, pipeline_specs, ignore_policies, splitters

    def _make_grid(
        self,
        n_specs: int = 2,
        n_policies: int = 2,
        n_splitters: int = 2,
        matching_part: str = "heads",
    ) -> tuple[SimilarityGrid, list[SimilarityRunSpec], list[MagicMock], list[MagicMock], list[MagicMock]]:
        matching_dataset = MagicMock(name="matching_dataset")
        matching_dataset.wildlife_dataset.metadata = "metadata"

        run_specs, pipeline_specs, ignore_policies, splitters = self._make_run_specs(n_specs, n_policies, n_splitters, matching_part)
        cache = MagicMock(name="cache")

        grid = SimilarityGrid(
            matching_dataset=matching_dataset,
            run_specs=run_specs,
            cache=cache,
            cache_prefix="prefix",
            device="cpu",
        )
        return grid, run_specs, pipeline_specs, ignore_policies, splitters

    def test_empty_run_specs_returns_without_error(self) -> None:
        grid = SimilarityGrid(
            matching_dataset=MagicMock(name="matching_dataset"),
            run_specs=[],
            cache=MagicMock(name="cache"),
            cache_prefix="prefix",
            device="cpu",
        )
        grid.run()

    def test_raises_when_run_specs_have_different_matching_parts(self) -> None:
        run_specs_heads, *_ = self._make_run_specs(1, 1, 1, matching_part="heads")
        run_specs_bodies, *_ = self._make_run_specs(1, 1, 1, matching_part="bodies")
        grid = SimilarityGrid(
            matching_dataset=MagicMock(name="matching_dataset"),
            run_specs=run_specs_heads + run_specs_bodies,
            cache=MagicMock(name="cache"),
            cache_prefix="prefix",
            device="cpu",
        )
        with pytest.raises(ValueError):
            grid.run()

    def test_every_pipeline_spec_is_built_once_with_device_and_cache_prefix(self) -> None:
        grid, run_specs, pipeline_specs, ignore_policies, splitters = self._make_grid()
        grid.run()
        for spec in pipeline_specs:
            spec.build.assert_called_once_with("cpu", "prefix")

    def test_each_splitter_get_split_called_once_with_metadata(self) -> None:
        grid, run_specs, pipeline_specs, ignore_policies, splitters = self._make_grid()
        grid.run()
        for splitter in splitters:
            splitter.get_split.assert_called_once_with("metadata")

    def test_runs_a_job_for_every_run_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid, run_specs, pipeline_specs, ignore_policies, splitters = self._make_grid(n_specs=3, n_policies=2, n_splitters=2)

        created_jobs: list[dict[str, Any]] = []
        original_init = SimilarityJob.__init__

        def spy_init(self: SimilarityJob, **kwargs: Any) -> None:
            created_jobs.append(kwargs)
            original_init(self, **kwargs)

        run_mock = MagicMock()
        monkeypatch.setattr(SimilarityJob, "__init__", spy_init)
        monkeypatch.setattr(SimilarityJob, "run", run_mock)

        grid.run()

        assert run_mock.call_count == len(run_specs)
        assert len(created_jobs) == len(run_specs)
        seen_run_spec_ids = {id(kw["run_spec"]) for kw in created_jobs}
        expected_run_spec_ids = {id(rs) for rs in run_specs}
        assert seen_run_spec_ids == expected_run_spec_ids
