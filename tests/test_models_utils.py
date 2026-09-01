from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from wildlife_datasets.datasets import WildlifeDataset

from animal_re_id.models.utils import get_index


def make_dataset_with_metadata(metadata: pd.DataFrame) -> WildlifeDataset:
    return cast(WildlifeDataset, SimpleNamespace(metadata=metadata))


class TestGetIndex:
    def test_returns_index_for_unique_match(self, capsys: pytest.CaptureFixture[str]) -> None:
        dataset = make_dataset_with_metadata(pd.DataFrame({"image_id": ["a", "b", "c"]}))
        assert get_index(dataset, "b") == 1

    def test_returns_none_and_prints_when_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        dataset = make_dataset_with_metadata(pd.DataFrame({"image_id": ["a", "b"]}))
        result = get_index(dataset, "missing")
        assert result is None
        captured = capsys.readouterr()
        assert "missing" in captured.out

    def test_raises_on_duplicate_image_id(self) -> None:
        dataset = make_dataset_with_metadata(pd.DataFrame({"image_id": ["a", "a"]}))
        with pytest.raises(ValueError, match="multiple times"):
            get_index(dataset, "a")

    def test_uses_original_dataframe_index_not_position(self) -> None:
        df = pd.DataFrame({"image_id": ["a", "b", "c"]}, index=[10, 20, 30])
        dataset = make_dataset_with_metadata(df)
        assert get_index(dataset, "b") == 20
