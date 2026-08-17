from collections.abc import Callable

import pandas as pd
import pytest
from wildlife_datasets.datasets import WildlifeDataset


def make_wildlife_dataset(df: pd.DataFrame, img_load: str = "full", root: str | None = None) -> WildlifeDataset:
    """Build a real, lightweight WildlifeDataset backed by an in-memory dataframe.

    Passing root=None and check_files=False avoids touching the filesystem,
    so this is safe to use as a stand-in wherever a WildlifeDataset is expected.
    """
    return WildlifeDataset(root=root, df=df, img_load=img_load, check_files=False)


@pytest.fixture
def simple_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": [f"img_{i}.jpg" for i in range(6)],
            "identity": ["a", "a", "b", "b", "c", "unknown"],
            "year": [2019, 2020, 2019, 2021, 2021, 2021],
        }
    )


@pytest.fixture
def simple_wildlife_dataset(simple_metadata: pd.DataFrame) -> WildlifeDataset:
    return make_wildlife_dataset(simple_metadata)


@pytest.fixture
def wildlife_dataset_factory() -> Callable[..., WildlifeDataset]:
    return make_wildlife_dataset
