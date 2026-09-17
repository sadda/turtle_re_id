import os
from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd
from wildlife_datasets import datasets as wd_datasets

from ..config import resolve_path

SCORE_THRESHOLD = 0.8
SCORE_MIN = 0.3
MAX_PER_ENCOUNTER = 10
MIN_PER_ENCOUNTER = 2


def combine_datasets(wildlife_datasets, encounter_prefixes=None, **kwargs) -> wd_datasets.WildlifeDataset:
    if encounter_prefixes is None:
        encounter_prefixes = range(len(wildlife_datasets))
    metadatas = []
    for wildlife_dataset, encounter_prefix in zip(wildlife_datasets, encounter_prefixes):
        metadata = wildlife_dataset.metadata.copy()
        if wildlife_dataset.root is not None:
            metadata["path"] = wildlife_dataset.root + "/" + metadata["path"]
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
) -> wd_datasets.TurtlewatchEgypt_New:

    dfs = []
    if encounter_prefixes is None:
        encounter_prefixes = range(len(roots))
    for root, encounter_prefix in zip(roots, encounter_prefixes):
        wildlife_dataset = wd_datasets.TurtlewatchEgypt_New(
            f"{master_root}/{root}",
            file_name=file_name,
            load_segmentation=load_segmentation,
        )
        wildlife_dataset.df["path"] = root.split("/")[-1] + "/" + wildlife_dataset.df["path"]
        wildlife_dataset.df["encounter_id"] = wildlife_dataset.df["encounter_id"].apply(lambda x: f"{encounter_prefix}_{x}")
        dfs.append(wildlife_dataset.df)
    df = pd.concat(dfs).reset_index(drop=True)
    return wd_datasets.TurtlewatchEgypt_New(root=master_root, df=df, **kwargs)


def combine_datasets_twe_from_paths(paths, **kwargs) -> wd_datasets.TurtlewatchEgypt_New:
    return combine_datasets_twe(
        resolve_path(paths["root_new"]),
        paths["roots_new"],
        resolve_path(paths["file_name_individuals"]),
        **kwargs,
    )


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


class MatchingDataset(ABC):
    def __init__(self, wildlife_dataset, matching_parts):
        self.wildlife_dataset = wildlife_dataset
        self.wildlife_dataset.metadata = self.wildlife_dataset.metadata.reset_index(drop=True)
        self.matching_parts = matching_parts

    @classmethod
    @abstractmethod
    def from_paths(cls, paths, *args, **kwargs):
        raise NotImplementedError

    def remove_unknown(self):
        mask = self.wildlife_dataset.metadata["identity"] != "unknown"
        self.wildlife_dataset = self.wildlife_dataset.get_subset(mask)

    def keep_only_unknown(self):
        mask = self.wildlife_dataset.metadata["identity"] == "unknown"
        self.wildlife_dataset = self.wildlife_dataset.get_subset(mask)

    def for_matching_part(self, matching_part):
        labels = self.matching_parts[matching_part]["labels"]
        mask = self.wildlife_dataset.metadata["label"].isin(labels)
        wildlife_dataset = self.wildlife_dataset.get_subset(mask)

        narrowed = deepcopy(self)
        narrowed.wildlife_dataset = wildlife_dataset
        return narrowed


def _build_matching_parts(front_labels, hind_labels, database_query_matching):
    cols_equal = [] if database_query_matching else ["identity", "encounter_id"]
    return {
        "heads": {"labels": ["head"], "cols_equal": cols_equal, "cols_unequal": []},
        "front flippers": {
            "labels": front_labels,
            "cols_equal": cols_equal,
            "cols_unequal": ["label", "orientation"],
        },
        "hind flippers": {
            "labels": hind_labels,
            "cols_equal": cols_equal,
            "cols_unequal": ["label", "orientation"],
        },
        "bodies": {"labels": ["turtle"], "cols_equal": cols_equal, "cols_unequal": []},
    }


class TurtleDetectorSegmented(MatchingDataset):
    def __init__(self, wildlife_dataset, database_query_matching):
        matching_parts = _build_matching_parts(["flipper_fl", "flipper_fr"], ["flipper_hl", "flipper_hr"], database_query_matching)
        super().__init__(wildlife_dataset, matching_parts)

    def select_subset(
        self,
        score_threshold=SCORE_THRESHOLD,
        score_min=SCORE_MIN,
        max_per_encounter=MAX_PER_ENCOUNTER,
        min_per_encounter=MIN_PER_ENCOUNTER,
    ):

        mask = []
        for _, df_label in self.wildlife_dataset.metadata.groupby("label"):
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

        self.wildlife_dataset = self.wildlife_dataset.get_subset(mask)


class TurtlesFlippersMerged(MatchingDataset):
    def __init__(self, wildlife_dataset, database_query_matching):
        matching_parts = _build_matching_parts(["front flipper"], ["hind flipper"], database_query_matching)
        super().__init__(wildlife_dataset, matching_parts)


class TurtlesOfSMSRC(TurtleDetectorSegmented):
    def __init__(self, root, is_database, dataset_kwargs=None, subset_kwargs=None):
        dataset_kwargs = dataset_kwargs or {}
        subset_kwargs = subset_kwargs or {}

        img_load = dataset_kwargs.pop("img_load", "bbox")
        load_label = dataset_kwargs.pop("load_label", True)

        wildlife_dataset = wd_datasets.TurtlesOfSMSRC(
            root,
            img_load=img_load,
            load_segmentation=True,
            load_label=load_label,
            **dataset_kwargs,
        )
        super().__init__(wildlife_dataset, is_database)
        self.select_subset(**subset_kwargs)

    @classmethod
    def from_paths(cls, paths, is_database, dataset_kwargs=None, subset_kwargs=None):
        return cls(resolve_path(paths["root"]), is_database, dataset_kwargs=dataset_kwargs, subset_kwargs=subset_kwargs)


class TurtlewatchEgypt_Citizen(TurtleDetectorSegmented):
    def __init__(self, root, is_database, dataset_kwargs=None, subset_kwargs=None):
        dataset_kwargs = dataset_kwargs or {}
        subset_kwargs = subset_kwargs or {}

        img_load = dataset_kwargs.pop("img_load", "bbox")
        load_label = dataset_kwargs.pop("load_label", True)

        wildlife_dataset = wd_datasets.TurtlewatchEgypt_Citizen(
            root,
            img_load=img_load,
            load_segmentation=True,
            load_label=load_label,
            **dataset_kwargs,
        )
        super().__init__(wildlife_dataset, is_database)
        self.select_subset(**subset_kwargs)

    @classmethod
    def from_paths(cls, paths, is_database, dataset_kwargs=None, subset_kwargs=None):
        return cls(resolve_path(paths["root_citizen"]), is_database, dataset_kwargs=dataset_kwargs, subset_kwargs=subset_kwargs)


class TurtlewatchEgypt_New(TurtleDetectorSegmented):
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

        wildlife_dataset = combine_datasets_twe(
            root_master,
            roots,
            file_name,
            img_load=img_load,
            load_segmentation=True,
            load_label=load_label,
            **dataset_kwargs,
        )
        wildlife_dataset.metadata["year"] = wildlife_dataset.metadata["path"].str[:4].astype(int)
        super().__init__(wildlife_dataset, is_database)
        self.select_subset(**subset_kwargs)

    @classmethod
    def from_paths(cls, paths, is_database, dataset_kwargs=None, subset_kwargs=None):
        return cls(
            resolve_path(paths["root_new"]),
            paths["roots_new"],
            is_database,
            resolve_path(paths["file_name_individuals"]),
            dataset_kwargs=dataset_kwargs,
            subset_kwargs=subset_kwargs,
        )


class SeaTurtleID2022(TurtlesFlippersMerged):
    def __init__(self, root, database_query_matching, dataset_kwargs=None):
        dataset_kwargs = dataset_kwargs or {}
        img_load = dataset_kwargs.pop("img_load", "bbox")
        load_label = dataset_kwargs.pop("load_label", True)

        metadata_h = wd_datasets.SeaTurtleID2022(root, category_name="head").metadata
        metadata_f = wd_datasets.SeaTurtleID2022(root, category_name="flipper").metadata
        metadata_t = wd_datasets.SeaTurtleID2022(root, category_name="turtle").metadata

        metadata_h["image_id"] = "h_" + metadata_h["image_id"].astype(str)
        metadata_h["label"] = "head"
        metadata_f["image_id"] = "f_" + metadata_f["image_id"].astype(str)
        metadata_t["image_id"] = "t_" + metadata_t["image_id"].astype(str)
        metadata_t["label"] = "turtle"

        mask = metadata_f["orientation"].astype(str).str.startswith("front")
        metadata_f.loc[mask, "label"] = "front flipper"
        mask = metadata_f["orientation"].astype(str).str.startswith("rear")
        metadata_f.loc[mask, "label"] = "hind flipper"

        metadata = pd.concat((metadata_h, metadata_f, metadata_t))
        dates = pd.to_datetime(metadata["date"])
        metadata["year"] = dates.apply(lambda x: x.year)
        metadata["encounter_id"] = metadata["identity"] + "_" + metadata["date"]

        wildlife_dataset = wd_datasets.SeaTurtleID2022(root, df=metadata, img_load=img_load, load_label=load_label)
        super().__init__(wildlife_dataset, database_query_matching)

    @classmethod
    def from_paths(cls, paths, database_query_matching, dataset_kwargs=None):
        return cls(resolve_path(paths["root"]), database_query_matching, dataset_kwargs=dataset_kwargs)


class TurtlewatchEgypt_Master(TurtlesFlippersMerged):
    def __init__(
        self,
        root_heads,
        root_flippers_front,
        root_flippers_hind,
        database_query_matching,
        file_name,
        dataset_kwargs=None,
        score_min=SCORE_THRESHOLD,
    ):
        dataset_kwargs = dataset_kwargs or {}
        load_label = dataset_kwargs.pop("load_label", True)

        metadata_h = wd_datasets.TurtlewatchEgypt_Master(root_heads, file_name=file_name, load_segmentation=True, **dataset_kwargs).metadata
        metadata_f = wd_datasets.TurtlewatchEgypt_Master(root_flippers_front, file_name=file_name, **dataset_kwargs).metadata
        metadata_r = wd_datasets.TurtlewatchEgypt_Master(root_flippers_hind, file_name=file_name, **dataset_kwargs).metadata

        convertion = {
            "flipper_fr": "front flipper",
            "flipper_fl": "front flipper",
            "flipper_hl": "hind flipper",
            "flipper_hr": "hind flipper",
        }
        metadata_h["label"] = metadata_h["label"].replace(convertion)
        metadata_h["path"] = metadata_h["path"].apply(lambda x: os.path.join(root_heads, x))
        metadata_h = metadata_h[~metadata_h["label"].isnull()]
        metadata_h = metadata_h[metadata_h["score"] >= score_min]
        metadata_f["label"] = "front flipper"
        metadata_f["path"] = metadata_f["path"].apply(lambda x: os.path.join(root_flippers_front, x))
        metadata_r["label"] = "hind flipper"
        metadata_r["path"] = metadata_r["path"].apply(lambda x: os.path.join(root_flippers_hind, x))

        metadata = pd.concat((metadata_h, metadata_f, metadata_r))
        wildlife_dataset = wd_datasets.TurtlewatchEgypt_Master(
            root=None,
            df=metadata,
            load_label=load_label,
            file_name=file_name,
            **dataset_kwargs,
        )
        super().__init__(wildlife_dataset, database_query_matching)

    @classmethod
    def from_paths(cls, paths, database_query_matching, dataset_kwargs=None, score_min=SCORE_THRESHOLD):
        return cls(
            resolve_path(paths["root_heads"]),
            resolve_path(paths["root_flippers_front"]),
            resolve_path(paths["root_flippers_hind"]),
            database_query_matching,
            resolve_path(paths["file_name_individuals"]),
            dataset_kwargs=dataset_kwargs,
            score_min=score_min,
        )


class TurtlewatchEgypt_Combined(MatchingDataset):
    def __init__(
        self,
        root_master,
        roots,
        root_heads,
        root_flippers_front,
        root_flippers_hind,
        file_name,
        dataset_kwargs=None,
        subset_kwargs=None,
    ):
        dataset_kwargs = dataset_kwargs or {}
        subset_kwargs = subset_kwargs or {}

        load_label = dataset_kwargs.pop("load_label", True)
        score_min = subset_kwargs.pop("score_min", SCORE_MIN)

        matching_dataset1 = TurtlewatchEgypt_New(
            root_master,
            roots,
            False,
            file_name,
            dataset_kwargs=dataset_kwargs,
            subset_kwargs=subset_kwargs,
        )
        matching_dataset2 = TurtlewatchEgypt_Master(
            root_heads,
            root_flippers_front,
            root_flippers_hind,
            False,
            file_name,
            dataset_kwargs=dataset_kwargs,
            score_min=score_min,
        )
        wildlife_dataset = combine_datasets(
            [matching_dataset1.wildlife_dataset, matching_dataset2.wildlife_dataset],
            load_label=load_label,
            **dataset_kwargs,
        )
        matching_parts = combine_matching_parts(matching_dataset1.matching_parts, matching_dataset2.matching_parts)
        super().__init__(wildlife_dataset, matching_parts)

    @classmethod
    def from_paths(cls, paths, dataset_kwargs=None, subset_kwargs=None):
        return cls(
            resolve_path(paths["root_new"]),
            paths["roots_new"],
            resolve_path(paths["root_heads"]),
            resolve_path(paths["root_flippers_front"]),
            resolve_path(paths["root_flippers_hind"]),
            resolve_path(paths["file_name_individuals"]),
            dataset_kwargs=dataset_kwargs,
            subset_kwargs=subset_kwargs,
        )


class StripeSpotter(MatchingDataset):
    def __init__(self, root, database_query_matching, dataset_kwargs=None):
        matching_parts = {
            "full": {"labels": ["full"], "cols_equal": [], "cols_unequal": []},
            "bodies": {"labels": ["body"], "cols_equal": [], "cols_unequal": []},
        }        

        dataset_kwargs = dataset_kwargs or {}
        load_label = dataset_kwargs.pop("load_label", True)

        metadata_bodies = wd_datasets.StripeSpotter(root).metadata
        metadata_bodies["encounter_id"] = range(len(metadata_bodies))
        metadata_bodies["year"] = 2011
        metadata_bodies["label"] = "body"
        metadata_full = metadata_bodies.copy()
        metadata_full["label"] = "full"
        metadata_full["bbox"] = None

        metadata = pd.concat((metadata_bodies, metadata_full))
        wildlife_dataset = wd_datasets.SeaTurtleID2022(root, df=metadata, img_load="bbox", load_label=load_label)
        
        super().__init__(wildlife_dataset, matching_parts)

    @classmethod
    def from_paths(cls, paths, database_query_matching, dataset_kwargs=None):
        return cls(resolve_path(paths["root"]), database_query_matching, dataset_kwargs=dataset_kwargs)

