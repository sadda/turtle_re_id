import os

import pandas as pd
from wildlife_datasets import datasets as wd_datasets

SCORE_THRESHOLD = 0.8
SCORE_MIN = 0.3
MAX_PER_ENCOUNTER = 10
MIN_PER_ENCOUNTER = 2


def combine_datasets(datasets, encounter_prefixes=None, **kwargs):
    if encounter_prefixes is None:
        encounter_prefixes = range(len(datasets))
    metadatas = []
    for dataset, encounter_prefix in zip(datasets, encounter_prefixes):
        metadata = dataset.metadata.copy()
        if dataset.root is not None:
            metadata["path"] = dataset.root + "/" + metadata["path"]
        if "encounter_id" in metadata.columns:
            mask = ~metadata["encounter_id"].isnull()
            new_encounter_id = metadata.loc[mask, "encounter_id"].apply(lambda x: f"{encounter_prefix}_{x}")
            metadata.loc[mask, "encounter_id"] = new_encounter_id
        metadatas.append(metadata)
    metadatas = pd.concat(metadatas)
    return wd_datasets.WildlifeDataset(root=None, df=metadatas, **kwargs)


def combine_datasets_twe(
    master_root,
    roots,
    file_name,
    load_segmentation=True,
    encounter_prefixes=None,
    **kwargs,
):

    dfs = []
    if encounter_prefixes is None:
        encounter_prefixes = range(len(roots))
    for root, encounter_prefix in zip(roots, encounter_prefixes):
        dataset = wd_datasets.TurtlewatchEgypt_New(
            f"{master_root}/{root}",
            file_name=file_name,
            load_segmentation=load_segmentation,
        )
        dataset.df["path"] = root.split("/")[-1] + "/" + dataset.df["path"]
        dataset.df["encounter_id"] = dataset.df["encounter_id"].apply(lambda x: f"{encounter_prefix}_{x}")
        dfs.append(dataset.df)
    df = pd.concat(dfs).reset_index(drop=True)
    return wd_datasets.TurtlewatchEgypt_New(root=master_root, df=df, **kwargs)


def combine_matching_parts(matching_parts1, matching_parts2):
    assert matching_parts1.keys() == matching_parts2.keys()
    keys1 = matching_parts1.keys()
    keys2 = ["labels", "cols_equal", "cols_unequal"]
    matching_parts = {key: {} for key in keys1}
    for key1 in keys1:
        for key2 in keys2:
            value1 = matching_parts1[key1][key2]
            value2 = matching_parts2[key1][key2]
            value = list(set(value1).union(set(value2)))
            matching_parts[key1][key2] = value
    return matching_parts


class Dataset:
    def __init__(self, dataset, matching_parts):
        self.dataset = dataset
        self.dataset.metadata = self.dataset.metadata.reset_index(drop=True)
        self.matching_parts = matching_parts

    def remove_unknown(self):
        mask = self.dataset.metadata["identity"] != "unknown"
        self.dataset = self.dataset.get_subset(mask)

    def keep_only_unknown(self):
        mask = self.dataset.metadata["identity"] == "unknown"
        self.dataset = self.dataset.get_subset(mask)


class TurtlesSegmented(Dataset):
    def __init__(self, dataset, database_query_matching):
        if database_query_matching:
            matching_parts = {
                "heads": {"labels": ["head"], "cols_equal": [], "cols_unequal": []},
                "front flippers": {
                    "labels": ["flipper_fl", "flipper_fr"],
                    "cols_equal": [],
                    "cols_unequal": ["label", "orientation"],
                },
                "rear flippers": {
                    "labels": ["flipper_rl", "flipper_rr"],
                    "cols_equal": [],
                    "cols_unequal": ["label", "orientation"],
                },
            }
        else:
            matching_parts = {
                "heads": {
                    "labels": ["head"],
                    "cols_equal": ["identity", "encounter_id"],
                    "cols_unequal": [],
                },
                "front flippers": {
                    "labels": ["flipper_fl", "flipper_fr"],
                    "cols_equal": ["identity", "encounter_id"],
                    "cols_unequal": ["label", "orientation"],
                },
                "rear flippers": {
                    "labels": ["flipper_rl", "flipper_rr"],
                    "cols_equal": ["identity", "encounter_id"],
                    "cols_unequal": ["label", "orientation"],
                },
            }
        super().__init__(dataset, matching_parts)

    def select_subset(
        self,
        score_threshold=SCORE_THRESHOLD,
        score_min=SCORE_MIN,
        max_per_encounter=MAX_PER_ENCOUNTER,
        min_per_encounter=MIN_PER_ENCOUNTER,
    ):

        mask = []
        for _, df_label in self.dataset.metadata.groupby("label"):
            for _, df_encounter in df_label.groupby("encounter_id"):
                df_encounter = df_encounter.sort_values("score", ascending=False)
                df_encounter = df_encounter.iloc[:max_per_encounter]
                df_encounter = df_encounter[df_encounter["score"] >= score_min]
                n_above = (df_encounter["score"] >= score_threshold).sum()
                if n_above >= min_per_encounter:
                    df_encounter = df_encounter.iloc[:n_above]
                else:
                    df_encounter = df_encounter.iloc[:min_per_encounter]
                mask.extend(df_encounter.index)

        self.dataset = self.dataset.get_subset(mask)


class TurtlesFlippersMerged(Dataset):
    def __init__(self, dataset, database_query_matching):
        if database_query_matching:
            matching_parts = {
                "heads": {"labels": ["head"], "cols_equal": [], "cols_unequal": []},
                "front flippers": {
                    "labels": ["front flipper"],
                    "cols_equal": [],
                    "cols_unequal": ["label", "orientation"],
                },
                "rear flippers": {
                    "labels": ["rear flipper"],
                    "cols_equal": [],
                    "cols_unequal": ["label", "orientation"],
                },
            }
        else:
            matching_parts = {
                "heads": {
                    "labels": ["head"],
                    "cols_equal": ["identity", "encounter_id"],
                    "cols_unequal": [],
                },
                "front flippers": {
                    "labels": ["front flipper"],
                    "cols_equal": ["identity", "encounter_id"],
                    "cols_unequal": ["label", "orientation"],
                },
                "rear flippers": {
                    "labels": ["rear flipper"],
                    "cols_equal": ["identity", "encounter_id"],
                    "cols_unequal": ["label", "orientation"],
                },
            }
        super().__init__(dataset, matching_parts)


class TurtlesOfSMSRC(TurtlesSegmented):
    def __init__(self, root, is_database, dataset_kwargs=None, subset_kwargs=None):
        dataset_kwargs = dataset_kwargs or {}
        subset_kwargs = subset_kwargs or {}

        img_load = dataset_kwargs.pop("img_load", "bbox")
        load_label = dataset_kwargs.pop("load_label", True)

        dataset = wd_datasets.TurtlesOfSMSRC(
            root,
            img_load=img_load,
            load_segmentation=True,
            load_label=load_label,
            **dataset_kwargs,
        )
        super().__init__(dataset, is_database)
        self.select_subset(**subset_kwargs)


class TurtlewatchEgypt_Citizen(TurtlesSegmented):
    def __init__(self, root, is_database, dataset_kwargs=None, subset_kwargs=None):
        dataset_kwargs = dataset_kwargs or {}
        subset_kwargs = subset_kwargs or {}

        img_load = dataset_kwargs.pop("img_load", "bbox")
        load_label = dataset_kwargs.pop("load_label", True)

        dataset = wd_datasets.TurtlewatchEgypt_Citizen(
            root,
            img_load=img_load,
            load_segmentation=True,
            load_label=load_label,
            **dataset_kwargs,
        )
        super().__init__(dataset, is_database)
        self.select_subset(**subset_kwargs)


class TurtlewatchEgypt_New(TurtlesSegmented):
    def __init__(
        self,
        root_master,
        roots,
        is_database,
        file_name,
        dataset_kwargs=None,
        subset_kwargs=None,
    ):
        dataset_kwargs = dataset_kwargs or {}
        subset_kwargs = subset_kwargs or {}

        img_load = dataset_kwargs.pop("img_load", "bbox")
        load_label = dataset_kwargs.pop("load_label", True)

        dataset = combine_datasets_twe(
            root_master,
            roots,
            file_name,
            img_load=img_load,
            load_segmentation=True,
            load_label=load_label,
            **dataset_kwargs,
        )
        super().__init__(dataset, is_database)
        self.select_subset(**subset_kwargs)


class SeaTurtleID2022(TurtlesFlippersMerged):
    def __init__(self, root, database_query_matching):
        metadata_h = wd_datasets.SeaTurtleID2022(root, category_name="head").metadata
        metadata_f = wd_datasets.SeaTurtleID2022(root, category_name="flipper").metadata

        metadata_h["image_id"] = "h_" + metadata_h["image_id"].astype(str)
        metadata_h["label"] = "head"
        metadata_f["image_id"] = "f_" + metadata_f["image_id"].astype(str)
        mask = metadata_f["orientation"].astype(str).str.startswith("front")
        metadata_f.loc[mask, "label"] = "front flipper"
        mask = metadata_f["orientation"].astype(str).str.startswith("rear")
        metadata_f.loc[mask, "label"] = "rear flipper"

        metadata = pd.concat((metadata_h, metadata_f))
        dates = pd.to_datetime(metadata["date"])
        metadata["year"] = dates.apply(lambda x: x.year)
        metadata["encounter_id"] = metadata["identity"] + "_" + metadata["date"]

        dataset = wd_datasets.SeaTurtleID2022(root, df=metadata, img_load="bbox", load_label=True)
        super().__init__(dataset, database_query_matching)


class TurtlewatchEgypt_Master(TurtlesFlippersMerged):
    def __init__(
        self,
        root_heads,
        root_flippers_f,
        root_flippers_r,
        database_query_matching,
        file_name,
        dataset_kwargs=None,
        score_min=SCORE_THRESHOLD,
    ):
        dataset_kwargs = dataset_kwargs or {}
        load_label = dataset_kwargs.pop("load_label", True)

        metadata_h = wd_datasets.TurtlewatchEgypt_Master(
            root_heads, file_name=file_name, load_segmentation=True, **dataset_kwargs
        ).metadata
        metadata_f = wd_datasets.TurtlewatchEgypt_Master(
            root_flippers_f, file_name=file_name, **dataset_kwargs
        ).metadata
        metadata_r = wd_datasets.TurtlewatchEgypt_Master(
            root_flippers_r, file_name=file_name, **dataset_kwargs
        ).metadata

        convertion = {
            "flipper_fr": "front flipper",
            "flipper_fl": "front flipper",
            "flipper_rl": "rear flipper",
            "flipper_rr": "rear flipper",
        }
        metadata_h["label"] = metadata_h["label"].replace(convertion)
        metadata_h["path"] = metadata_h["path"].apply(lambda x: os.path.join(root_heads, x))
        metadata_h = metadata_h[~metadata_h["label"].isnull()]
        metadata_h = metadata_h[metadata_h["score"] >= score_min]
        metadata_f["label"] = "front flipper"
        metadata_f["path"] = metadata_f["path"].apply(lambda x: os.path.join(root_flippers_f, x))
        metadata_r["label"] = "rear flipper"
        metadata_r["path"] = metadata_r["path"].apply(lambda x: os.path.join(root_flippers_r, x))

        metadata = pd.concat((metadata_h, metadata_f, metadata_r))
        dataset = wd_datasets.TurtlewatchEgypt_Master(
            root=None,
            df=metadata,
            load_label=load_label,
            file_name=file_name,
            **dataset_kwargs,
        )
        super().__init__(dataset, database_query_matching)


class TurtlewatchEgypt_Combined(Dataset):
    def __init__(
        self,
        root_master,
        roots,
        root_heads,
        root_flippers_f,
        root_flippers_r,
        file_name,
        dataset_kwargs=None,
        subset_kwargs=None,
    ):
        dataset_kwargs = dataset_kwargs or {}
        subset_kwargs = subset_kwargs or {}

        load_label = dataset_kwargs.pop("load_label", True)
        score_min = subset_kwargs.pop("score_min", SCORE_MIN)

        dataset1 = TurtlewatchEgypt_New(
            root_master,
            roots,
            False,
            file_name,
            dataset_kwargs=dataset_kwargs,
            subset_kwargs=subset_kwargs,
        )
        dataset2 = TurtlewatchEgypt_Master(
            root_heads,
            root_flippers_f,
            root_flippers_r,
            False,
            file_name,
            dataset_kwargs=dataset_kwargs,
            score_min=score_min,
        )
        dataset = combine_datasets(
            [dataset1.dataset, dataset2.dataset],
            load_label=load_label,
            **dataset_kwargs,
        )
        matching_parts = combine_matching_parts(dataset1.matching_parts, dataset2.matching_parts)
        super().__init__(dataset, matching_parts)
