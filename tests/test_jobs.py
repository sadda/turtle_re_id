from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from animal_re_id.similarity.jobs import SimilarityGrid, SimilarityJob


def make_job(cache_exists: bool = False) -> SimilarityJob:
    matching_dataset = MagicMock(name="matching_dataset")
    wildlife_dataset = MagicMock(name="wildlife_dataset")
    wildlife_dataset.img_load = "bbox"

    pipeline_spec = MagicMock(name="pipeline_spec")
    pipeline_spec.name = "dino"
    pipeline_spec.B = 5
    pipeline_spec.compute.return_value = np.array([[1.0]])

    ignore_policy = MagicMock(name="ignore_policy")
    ignore_policy.cache_key = "allow_all"

    split = MagicMock(name="split")
    split.name = "2021"

    cache = MagicMock(name="cache")
    cache.key_for.return_value = ["key"]
    cache.exists.return_value = cache_exists

    job = SimilarityJob(
        matching_dataset=matching_dataset,
        wildlife_dataset=wildlife_dataset,
        matching_part="heads",
        pipeline_spec=pipeline_spec,
        ignore_policy=ignore_policy,
        split=split,
        cache=cache,
    )
    return job


class TestSimilarityJob:
    def test_run_skips_computation_when_cached(self) -> None:
        job = make_job(cache_exists=True)
        job.run()
        job.split.database.assert_not_called()
        job.split.query.assert_not_called()
        job.pipeline_spec.compute.assert_not_called()
        job.cache.save.assert_not_called()

    def test_run_computes_and_saves_when_not_cached(self) -> None:
        job = make_job(cache_exists=False)
        job.split.database.return_value = "database_subset"
        job.split.query.return_value = "query_subset"
        job.ignore_policy.get_pair_selector.return_value = "pair_selector"

        job.run()

        job.pipeline_spec.compute.assert_called_once_with("query_subset", "database_subset", "pair_selector")
        job.cache.save.assert_called_once()
        saved_key, saved_data = job.cache.save.call_args[0]
        assert saved_key == ["key"]
        assert saved_data["name"] == "dino"
        assert saved_data["B"] == 5
        assert saved_data["img_load"] == "bbox"
        assert saved_data["matching_part"] == "heads"
        assert saved_data["match_opposite_sides"] == "allow_all"
        assert saved_data["split_year"] == "2021"
        assert np.array_equal(saved_data["similarity"], np.array([[1.0]]))

    def test_run_uses_cache_key_for_with_correct_args(self) -> None:
        job = make_job(cache_exists=True)
        job.run()
        job.cache.key_for.assert_called_once_with(job.pipeline_spec, job.ignore_policy, job.split, "bbox", "heads")

    def test_run_builds_pair_selector_with_matching_dataset_and_part(self) -> None:
        job = make_job(cache_exists=False)
        job.split.database.return_value = "database_subset"
        job.split.query.return_value = "query_subset"

        job.run()

        job.ignore_policy.get_pair_selector.assert_called_once_with(job.matching_dataset, "heads")


class TestSimilarityGrid:
    def _make_grid(
        self,
        n_splits: int = 2,
        n_specs: int = 2,
        n_policies: int = 2,
    ) -> tuple[SimilarityGrid, list[MagicMock], list[MagicMock], list[MagicMock]]:
        wildlife_dataset = MagicMock(name="wildlife_dataset")
        wildlife_dataset.metadata = "metadata"

        splitter = MagicMock(name="splitter")
        splits = [MagicMock(name=f"split{i}") for i in range(n_splits)]
        splitter.get_splits.return_value = splits

        pipeline_specs = [MagicMock(name=f"spec{i}") for i in range(n_specs)]
        ignore_policies = [MagicMock(name=f"policy{i}") for i in range(n_policies)]

        cache = MagicMock(name="cache")

        grid = SimilarityGrid(
            matching_dataset=MagicMock(name="matching_dataset"),
            wildlife_dataset=wildlife_dataset,
            matching_part="heads",
            splitter=splitter,
            pipeline_specs=pipeline_specs,
            ignore_pairs_policies=ignore_policies,
            cache=cache,
            device="cpu",
            cache_prefix="prefix",
        )
        return grid, splits, pipeline_specs, ignore_policies

    def test_get_splits_called_with_metadata(self) -> None:
        grid, splits, pipeline_specs, ignore_policies = self._make_grid()
        grid.run()
        grid.splitter.get_splits.assert_called_once_with("metadata")

    def test_every_pipeline_spec_is_built_with_device_and_cache_prefix(self) -> None:
        grid, splits, pipeline_specs, ignore_policies = self._make_grid()
        grid.run()
        for spec in pipeline_specs:
            spec.build.assert_called_once_with("cpu", "prefix")

    def test_runs_a_job_for_every_combination(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid, splits, pipeline_specs, ignore_policies = self._make_grid(n_splits=2, n_specs=3, n_policies=2)

        created_jobs: list[dict[str, Any]] = []
        original_init = SimilarityJob.__init__

        def spy_init(self: SimilarityJob, **kwargs: Any) -> None:
            created_jobs.append(kwargs)
            original_init(self, **kwargs)

        run_mock = MagicMock()
        monkeypatch.setattr(SimilarityJob, "__init__", spy_init)
        monkeypatch.setattr(SimilarityJob, "run", run_mock)

        grid.run()

        assert run_mock.call_count == 2 * 3 * 2
        assert len(created_jobs) == 2 * 3 * 2
        seen_combinations = {(kw["pipeline_spec"], kw["ignore_policy"], kw["split"]) for kw in created_jobs}
        expected_combinations = {(spec, policy, split) for spec in pipeline_specs for policy in ignore_policies for split in splits}
        assert seen_combinations == expected_combinations

    def test_empty_pipeline_specs_runs_no_jobs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid, splits, pipeline_specs, ignore_policies = self._make_grid(n_specs=0)
        run_mock = MagicMock()
        monkeypatch.setattr(SimilarityJob, "run", run_mock)
        grid.run()
        run_mock.assert_not_called()
