from unittest.mock import MagicMock

import pytest

import animal_re_id.models.detector as detector_module
from animal_re_id.models.detector import turtle_detector


class TestTurtleDetector:
    def test_downloads_model_and_loads_yolo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_download = MagicMock(return_value="/cache/turtle_detector.pt")
        fake_yolo = MagicMock(return_value="yolo_model")
        monkeypatch.setattr(detector_module, "hf_hub_download", fake_download)
        monkeypatch.setattr(detector_module, "YOLO", fake_yolo)

        result = turtle_detector()

        fake_download.assert_called_once_with(repo_id="BVRA/TurtleDetector", filename="turtle_detector.pt")
        fake_yolo.assert_called_once_with("/cache/turtle_detector.pt")
        assert result == "yolo_model"

    def test_custom_repo_and_filename_are_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_download = MagicMock(return_value="/cache/custom.pt")
        fake_yolo = MagicMock(return_value="yolo_model")
        monkeypatch.setattr(detector_module, "hf_hub_download", fake_download)
        monkeypatch.setattr(detector_module, "YOLO", fake_yolo)

        turtle_detector(repo_id="Org/Model", filename="custom.pt")

        fake_download.assert_called_once_with(repo_id="Org/Model", filename="custom.pt")
        fake_yolo.assert_called_once_with("/cache/custom.pt")
