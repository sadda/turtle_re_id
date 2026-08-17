from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
from PIL import Image
from PIL.Image import Image as PILImage
from wildlife_datasets.datasets import WildlifeDataset

from animal_re_id.datasets.datasets import MatchingDataset
from animal_re_id.datasets.utils import add_date_exif, get_wildlife_dataset


class TestGetWildlifeDataset:
    def test_returns_subset_matching_labels(self) -> None:
        metadata = pd.DataFrame({"label": ["head", "flipper_fl", "turtle", "head"]})
        wildlife_dataset = SimpleNamespace(metadata=metadata)
        subset_calls: dict[str, pd.Series] = {}

        def get_subset(mask: pd.Series) -> WildlifeDataset:
            subset_calls["mask"] = mask
            return cast(WildlifeDataset, SimpleNamespace(metadata=metadata[mask]))

        wildlife_dataset.get_subset = get_subset
        matching_dataset = cast(
            MatchingDataset,
            SimpleNamespace(
                matching_parts={"heads": {"labels": ["head"]}},
                wildlife_dataset=wildlife_dataset,
            ),
        )

        result = get_wildlife_dataset(matching_dataset, "heads")
        assert list(subset_calls["mask"]) == [True, False, False, True]
        assert list(result.metadata["label"]) == ["head", "head"]

    def test_supports_multiple_labels_for_a_matching_part(self) -> None:
        metadata = pd.DataFrame({"label": ["flipper_fl", "flipper_fr", "flipper_hl"]})
        wildlife_dataset = SimpleNamespace(metadata=metadata)
        wildlife_dataset.get_subset = lambda mask: SimpleNamespace(metadata=metadata[mask])
        matching_dataset = cast(
            MatchingDataset,
            SimpleNamespace(
                matching_parts={"front flippers": {"labels": ["flipper_fl", "flipper_fr"]}},
                wildlife_dataset=wildlife_dataset,
            ),
        )

        result = get_wildlife_dataset(matching_dataset, "front flippers")
        assert list(result.metadata["label"]) == ["flipper_fl", "flipper_fr"]

    def test_no_matching_labels_returns_empty(self) -> None:
        metadata = pd.DataFrame({"label": ["turtle"]})
        wildlife_dataset = SimpleNamespace(metadata=metadata)
        wildlife_dataset.get_subset = lambda mask: SimpleNamespace(metadata=metadata[mask])
        matching_dataset = cast(
            MatchingDataset,
            SimpleNamespace(
                matching_parts={"heads": {"labels": ["head"]}},
                wildlife_dataset=wildlife_dataset,
            ),
        )

        result = get_wildlife_dataset(matching_dataset, "heads")
        assert len(result.metadata) == 0


class FakeImageDataset:
    def __init__(self, df: pd.DataFrame, image: PILImage) -> None:
        self.df = df
        self._image = image

    def __getitem__(self, i: int) -> PILImage:
        return self._image


class TestAddDateExif:
    def test_writes_exif_date_tags(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"date": ["2021-05-17T10:30:00"]})
        image = Image.new("RGB", (4, 4))
        dataset = FakeImageDataset(df, image)
        out_file = tmp_path / "out.jpg"

        add_date_exif(dataset, 0, str(out_file))

        assert out_file.exists()
        saved = Image.open(out_file)
        exif = saved.getexif()
        assert exif[306] == "2021:05:17 10:30:00"
        assert exif[36867] == "2021:05:17 10:30:00"
        assert exif[36868] == "2021:05:17 10:30:00"

    def test_uses_row_matching_given_index(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"date": ["2020-01-01T00:00:00", "2022-12-31T23:59:59"]})
        image = Image.new("RGB", (2, 2))
        dataset = FakeImageDataset(df, image)
        out_file = tmp_path / "out2.jpg"

        add_date_exif(dataset, 1, str(out_file))

        exif = Image.open(out_file).getexif()
        assert exif[306] == "2022:12:31 23:59:59"
