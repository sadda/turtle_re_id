from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
from PIL import Image
from wildlife_datasets.datasets import SeaTurtleID2022 as wd_SeaTurtleID2022
from wildlife_datasets.datasets import WildlifeDataset

from animal_re_id.config import resolve_path
from animal_re_id.datasets import TurtlewatchEgypt_New, combine_datasets_twe_from_paths


class Analysis(ABC):
    def __init__(self, paths) -> None:
        self.dataset: WildlifeDataset

        self.paths = paths
        self._load_dataset()        

    @abstractmethod
    def _load_dataset(self) -> None:
        pass

    @abstractmethod
    def get_summary(self) -> pd.DataFrame:
        pass

    def make_prediction(self, model, i):
        path = self.dataset.metadata["path"].iloc[i]
        file_name = self.dataset.get_absolute_path(path)

        img_array = model.predict(source=file_name, device="-1")[0].plot()
        img = Image.fromarray(img_array)

        return img


class SeaTurtleID2022(Analysis):
    def _get_counts(self, dataset):
        counts = defaultdict(int)
        for key, df in dataset.metadata.groupby(["identity", "date"]):
            counts[key] += len(df)
        return counts

    def _load_dataset(self) -> None:
        root = resolve_path(self.paths["root"])
        self.dataset = wd_SeaTurtleID2022(root)
        self.dataset_heads = wd_SeaTurtleID2022(root, category_name="head")
        dataset_flippers = wd_SeaTurtleID2022(root, category_name="flipper")

        mask1 = ~dataset_flippers.metadata["orientation"].isnull()
        mask2 = dataset_flippers.metadata["orientation"].str.startswith("front")
        mask3 = dataset_flippers.metadata["orientation"].str.startswith("rear")

        self.dataset_front = dataset_flippers.get_subset(mask1 & mask2)
        self.dataset_rear = dataset_flippers.get_subset(mask1 & mask3)

    def get_summary(self) -> pd.DataFrame:
        counts_h = self._get_counts(self.dataset_heads)
        counts_f = self._get_counts(self.dataset_front)
        counts_r = self._get_counts(self.dataset_rear)
        keys = set().union(counts_h, counts_f, counts_r)

        summary = {}
        for key in keys:
            identity, date = key
            summary[f"{identity}_{date}"] = {
                "identity": identity,
                "year": int(date[:4]),
                "heads": counts_h[key],
                "front flippers": counts_f[key],
                "hind flippers": counts_r[key],
                "total": counts_h[key] + counts_f[key] + counts_r[key],
            }
        return pd.DataFrame(summary).T


class TurtlewatchEgypt(Analysis):
    def _load_dataset(self) -> None:
        self.dataset = combine_datasets_twe_from_paths(self.paths, load_segmentation=False)
        mask1 = self.dataset.metadata["identity"] != "unknown"
        mask2 = ~self.dataset.metadata["identity"].isnull()
        self.dataset = self.dataset.get_subset(mask1 & mask2)
        self.dataset.metadata["year"] = self.dataset.metadata["path"].str[:4].astype(int)

    def get_summary(self) -> pd.DataFrame:
        encounters_id = self.dataset.metadata["encounter_id"].unique().tolist()
        summary = {encounter_id: {} for encounter_id in encounters_id}
        for encounter_id, df_encounter in self.dataset.metadata.groupby("encounter_id"):
            identity = df_encounter["identity"].unique().tolist()
            year = df_encounter["year"].unique().tolist()
            for arr in [identity, year]:
                if len(arr) != 1:
                    raise ValueError(f"Multiple values for encounter: {arr}")
            summary[encounter_id]["identity"] = identity[0]
            summary[encounter_id]["year"] = year[0]

        matching_dataset = TurtlewatchEgypt_New.from_paths(self.paths, True)
        metadata = matching_dataset.wildlife_dataset.metadata
        assert metadata["segmentation"].isnull().sum() == 0

        head_labels = matching_dataset.matching_parts["heads"]["labels"]
        front_labels = matching_dataset.matching_parts["front flippers"]["labels"]
        hind_labels = matching_dataset.matching_parts["hind flippers"]["labels"]

        for encounter_id, df_encounter in metadata.groupby("encounter_id"):
            if encounter_id in encounters_id:
                n_heads = (df_encounter["label"].isin(head_labels)).sum()
                n_front_flippers = (df_encounter["label"].isin(front_labels)).sum()
                n_hind_flippers = (df_encounter["label"].isin(hind_labels)).sum()

                summary[encounter_id]["heads"] = n_heads
                summary[encounter_id]["front flippers"] = n_front_flippers
                summary[encounter_id]["hind flippers"] = n_hind_flippers
                summary[encounter_id]["total"] = n_heads + n_front_flippers + n_hind_flippers

        summary_df = pd.DataFrame(summary).T
        summary_df = summary_df.replace(pd.NA, 0)
        return summary_df
