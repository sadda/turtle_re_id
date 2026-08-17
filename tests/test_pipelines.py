from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray
from wildlife_tools.similarity.calibration import IsotonicCalibration

import animal_re_id.models.pipelines as pipelines_module
from animal_re_id.models.pipelines import Wildfusion_Similarity, get_calibration


class TestGetCalibration:
    def test_returns_none_when_disabled(self) -> None:
        assert get_calibration(False) is None

    def test_returns_isotonic_calibration_when_enabled(self) -> None:
        calibration = get_calibration(True)
        assert isinstance(calibration, IsotonicCalibration)

    def test_returns_new_instance_each_call(self) -> None:
        assert get_calibration(True) is not get_calibration(True)


class TestWildfusionSimilarity:
    def _patch_wildfusion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        similarity_return: NDArray[np.number],
    ) -> tuple[MagicMock, MagicMock]:
        fake_wildfusion_instance = MagicMock()
        fake_wildfusion_instance.return_value = similarity_return
        fake_wildfusion_cls = MagicMock(return_value=fake_wildfusion_instance)
        monkeypatch.setattr(pipelines_module, "WildFusion", fake_wildfusion_cls)
        return fake_wildfusion_cls, fake_wildfusion_instance

    def test_init_builds_wildfusion_with_priority_and_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_cls, _ = self._patch_wildfusion(monkeypatch, np.array([[1.0]]))
        priority = object()
        pipeline = object()
        Wildfusion_Similarity(priority, pipeline, B=5)

        _, kwargs = fake_cls.call_args
        assert kwargs["calibrated_pipelines"] == [pipeline]
        assert kwargs["priority_pipeline"] is priority
        assert kwargs["pair_selector"] is None

    def test_init_passes_through_optional_pair_selector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_cls, _ = self._patch_wildfusion(monkeypatch, np.array([[1.0]]))
        pair_selector = object()
        Wildfusion_Similarity("priority", "pipeline", B=5, pair_selector=pair_selector)
        _, kwargs = fake_cls.call_args
        assert kwargs["pair_selector"] is pair_selector

    def test_call_invokes_wildfusion_with_B_and_casts_to_float64(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, fake_instance = self._patch_wildfusion(monkeypatch, np.array([[1, 2], [3, 4]], dtype=np.int32))
        sim = Wildfusion_Similarity("priority", "pipeline", B=7)

        result = sim("query", "database")

        fake_instance.assert_called_once_with("query", "database", B=7)
        assert result.dtype == np.float64
        assert np.array_equal(result, np.array([[1, 2], [3, 4]]))

    def test_set_pair_selector_updates_wildfusion_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, fake_instance = self._patch_wildfusion(monkeypatch, np.array([[1.0]]))
        sim = Wildfusion_Similarity("priority", "pipeline", B=1)
        new_selector = object()

        sim.set_pair_selector(new_selector)

        assert sim.wildfusion.pair_selector is new_selector
