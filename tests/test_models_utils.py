from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from wildlife_datasets.datasets import WildlifeDataset

from animal_re_id.models.utils import extract_column, gather_df, get_index


class TestExtractColumn:
    def test_returns_column_values_when_present(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = extract_column(df, "a")
        assert list(result) == [1, 2, 3]

    def test_returns_none_array_when_missing(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = extract_column(df, "missing")
        assert len(result) == 3
        assert all(v is None for v in result)

    def test_returns_numpy_array(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        assert isinstance(extract_column(df, "a"), np.ndarray)
        assert isinstance(extract_column(df, "missing"), np.ndarray)


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


class TestGatherDf:
    def _make_dataset(self, image_ids: list[str], identities: list[str], encounters: list[str]) -> WildlifeDataset:
        return cast(
            WildlifeDataset,
            SimpleNamespace(
                df=pd.DataFrame(
                    {
                        "image_id": image_ids,
                        "identity": identities,
                        "encounter_id": encounters,
                    }
                )
            ),
        )

    def test_gathers_rows_in_sorted_order(self) -> None:
        dataset_q = self._make_dataset(["q0", "q1"], ["ida", "idb"], ["eq0", "eq1"])
        dataset_d = self._make_dataset(["d0", "d1"], ["idc", "idd"], ["ed0", "ed1"])
        similarity = np.array([[0.1, 0.9], [0.5, 0.2]])
        i_sort: NDArray[np.intp] = np.array([0, 0])
        j_sort: NDArray[np.intp] = np.array([1, 0])

        result = gather_df(dataset_q, dataset_d, i_sort, j_sort, similarity, name="dino", matching_part="heads")

        assert list(result["image_id1"]) == ["q0", "q0"]
        assert list(result["image_id2"]) == ["d1", "d0"]
        assert list(result["encounter1"]) == ["eq0", "eq0"]
        assert list(result["encounter2"]) == ["ed1", "ed0"]
        assert list(result["identity1"]) == ["ida", "ida"]
        assert list(result["identity2"]) == ["idd", "idc"]
        assert list(result["matcher"]) == ["dino", "dino"]
        assert list(result["matching_part"]) == ["heads", "heads"]

    def test_similarity_orig_matches_raw_values(self) -> None:
        dataset_q = self._make_dataset(["q0"], ["ida"], ["eq0"])
        dataset_d = self._make_dataset(["d0", "d1"], ["idc", "idd"], ["ed0", "ed1"])
        similarity = np.array([[0.2, 0.8]])
        i_sort: NDArray[np.intp] = np.array([0, 0])
        j_sort: NDArray[np.intp] = np.array([1, 0])

        result = gather_df(dataset_q, dataset_d, i_sort, j_sort, similarity, name="mega", matching_part="bodies")
        assert list(result["similarity_orig"]) == pytest.approx([0.8, 0.2])

    def test_similarity_is_normalized_by_max(self) -> None:
        dataset_q = self._make_dataset(["q0"], ["ida"], ["eq0"])
        dataset_d = self._make_dataset(["d0"], ["idc"], ["ed0"])
        similarity = np.array([[0.5]])
        result = gather_df(dataset_q, dataset_d, np.array([0]), np.array([0]), similarity, name="mega", matching_part="bodies")
        assert result["similarity"].iloc[0] == pytest.approx(1.0)

    def test_missing_optional_columns_become_none(self) -> None:
        dataset_q = cast(WildlifeDataset, SimpleNamespace(df=pd.DataFrame({"image_id": ["q0"]})))
        dataset_d = cast(WildlifeDataset, SimpleNamespace(df=pd.DataFrame({"image_id": ["d0"]})))
        similarity = np.array([[1.0]])
        result = gather_df(dataset_q, dataset_d, np.array([0]), np.array([0]), similarity, name="mega", matching_part="bodies")
        assert result["identity1"].iloc[0] is None
        assert result["encounter1"].iloc[0] is None
