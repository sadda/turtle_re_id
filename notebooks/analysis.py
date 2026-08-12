from collections import defaultdict

import pandas as pd
from wildlife_datasets.datasets import SeaTurtleID2022 as wd_SeaTurtleID2022

from animal_re_id.config import resolve_path
from animal_re_id.datasets import TurtlewatchEgypt_New, combine_datasets_twe_from_paths


class SeaTurtleID2022:
    def __init__(self, paths):
        self.paths = paths

    def _get_counts(self, dataset):
        counts = defaultdict(int)
        for key, df in dataset.metadata.groupby(["identity", "date"]):
            counts[key] += len(df)
        return counts

    def get_summary(self):
        root = resolve_path(self.paths["root"])
        dataset_heads = wd_SeaTurtleID2022(root, category_name="head")
        dataset_flippers = wd_SeaTurtleID2022(root, category_name="flipper")

        mask1 = ~dataset_flippers.metadata["orientation"].isnull()
        mask2 = dataset_flippers.metadata["orientation"].str.startswith("front")
        mask3 = dataset_flippers.metadata["orientation"].str.startswith("rear")
        
        dataset_front = dataset_flippers.get_subset(mask1 & mask2)
        dataset_rear = dataset_flippers.get_subset(mask1 & mask3)

        counts_h = self._get_counts(dataset_heads)
        counts_f = self._get_counts(dataset_front)
        counts_r = self._get_counts(dataset_rear)
        keys = set().union(counts_h, counts_f, counts_r)        

        summary = {}
        for key in keys:
            identity, date = key
            summary[f"{identity}_{date}"] = {
                "identity": identity,
                "species": "cc",
                "year": int(date[:4]),
                "heads": counts_h[key],
                "front flippers": counts_f[key],
                "rear flippers": counts_r[key],
                "total": counts_h[key] + counts_f[key] + counts_r[key],
            }
        return pd.DataFrame(summary).T


class TurtlewatchEgypt:
    def __init__(self, paths):
        self.paths = paths

    def get_summary(self):
        dataset_true = combine_datasets_twe_from_paths(self.paths, load_segmentation=False)
        mask1 = dataset_true.metadata["identity"] != "unknown"
        mask2 = ~dataset_true.metadata["identity"].isnull()
        dataset_true = dataset_true.get_subset(mask1 & mask2)
        dataset_true.metadata["year"] = dataset_true.metadata["path"].str[:4].astype(int)

        encounters_id = dataset_true.metadata["encounter_id"].unique().tolist()
        summary = {encounter_id: {} for encounter_id in encounters_id}
        for encounter_id, df_encounter in dataset_true.metadata.groupby("encounter_id"):
            identity = df_encounter["identity"].unique().tolist()
            year = df_encounter["year"].unique().tolist()
            species = df_encounter["species"].unique().tolist()
            for arr in [identity, year, species]:
                if len(arr) != 1:
                    raise ValueError(f"Multiple values for encounter: {arr}")
            summary[encounter_id]["identity"] = identity[0]
            summary[encounter_id]["species"] = species[0]
            summary[encounter_id]["year"] = year[0]

        dataset = TurtlewatchEgypt_New.from_paths(self.paths, True)
        metadata = dataset.dataset.metadata
        assert metadata["segmentation"].isnull().sum() == 0

        for encounter_id, df_encounter in metadata.groupby("encounter_id"):
            if encounter_id in encounters_id:
                n_heads = (df_encounter["label"] == "head").sum()
                n_front_flippers = (df_encounter["label"].isin(("flipper_fl", "flipper_fr"))).sum()
                n_rear_flippers = (df_encounter["label"].isin(("flipper_rl", "flipper_rr"))).sum()

                summary[encounter_id]["heads"] = n_heads
                summary[encounter_id]["front flippers"] = n_front_flippers
                summary[encounter_id]["rear flippers"] = n_rear_flippers
                summary[encounter_id]["total"] = n_heads + n_front_flippers + n_rear_flippers

        summary_df = pd.DataFrame(summary).T
        summary_df = summary_df.replace(pd.NA, 0)
        return summary_df