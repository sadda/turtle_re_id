from pathlib import Path

import pandas as pd
from PIL import Image
from PIL.Image import Image as PILImage

from animal_re_id.datasets.datasets import MatchingDataset
from animal_re_id.datasets.utils import add_date_exif


class FakeWildlifeDataset:
    def __init__(self, metadata: pd.DataFrame) -> None:
        self.metadata = metadata

    def get_subset(self, mask: pd.Series) -> "FakeWildlifeDataset":
        return FakeWildlifeDataset(self.metadata[mask].reset_index(drop=True))


class FakeMatchingDataset(MatchingDataset):
    @classmethod
    def from_paths(cls, paths, *args, **kwargs):
        raise NotImplementedError


class TestForMatchingPart:
    def test_returns_subset_matching_labels(self) -> None:
        metadata = pd.DataFrame({"label": ["head", "flipper_fl", "turtle", "head"]})
        matching_dataset = FakeMatchingDataset(FakeWildlifeDataset(metadata), {"heads": {"labels": ["head"]}})

        result = matching_dataset.for_matching_part("heads")

        assert isinstance(result, FakeMatchingDataset)
        assert list(result.wildlife_dataset.metadata["label"]) == ["head", "head"]

    def test_supports_multiple_labels_for_a_matching_part(self) -> None:
        metadata = pd.DataFrame({"label": ["flipper_fl", "flipper_fr", "flipper_hl"]})
        matching_dataset = FakeMatchingDataset(FakeWildlifeDataset(metadata), {"front flippers": {"labels": ["flipper_fl", "flipper_fr"]}})

        result = matching_dataset.for_matching_part("front flippers")
        assert list(result.wildlife_dataset.metadata["label"]) == ["flipper_fl", "flipper_fr"]

    def test_no_matching_labels_returns_empty(self) -> None:
        metadata = pd.DataFrame({"label": ["turtle"]})
        matching_dataset = FakeMatchingDataset(FakeWildlifeDataset(metadata), {"heads": {"labels": ["head"]}})

        result = matching_dataset.for_matching_part("heads")
        assert len(result.wildlife_dataset.metadata) == 0

    def test_original_matching_dataset_is_unaffected(self) -> None:
        metadata = pd.DataFrame({"label": ["head", "turtle"]})
        matching_dataset = FakeMatchingDataset(FakeWildlifeDataset(metadata), {"heads": {"labels": ["head"]}})

        matching_dataset.for_matching_part("heads")
        assert len(matching_dataset.wildlife_dataset.metadata) == 2


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
