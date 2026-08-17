from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from wildlife_datasets.datasets import WildlifeDataset

import animal_re_id.datasets.datasets as datasets_module
from animal_re_id.datasets.datasets import (
    MatchingDataset,
    TurtleDetectorSegmented,
    TurtlesFlippersMerged,
    TurtlesOfSMSRC,
    TurtlewatchEgypt_Citizen,
    TurtlewatchEgypt_Combined,
    TurtlewatchEgypt_Master,
    TurtlewatchEgypt_New,
    _build_matching_parts,
    combine_datasets,
    combine_datasets_twe,
    combine_datasets_twe_from_paths,
    combine_matching_parts,
)


def make_wd(df: pd.DataFrame, root: str | None = None) -> WildlifeDataset:
    return WildlifeDataset(root=root, df=df, check_files=False)


class ConcreteSegmented(TurtleDetectorSegmented):
    @classmethod
    def from_paths(cls, paths: dict[str, Any], *args: Any, **kwargs: Any) -> "ConcreteSegmented":
        raise NotImplementedError


class ConcreteFlippersMerged(TurtlesFlippersMerged):
    @classmethod
    def from_paths(cls, paths: dict[str, Any], *args: Any, **kwargs: Any) -> "ConcreteFlippersMerged":
        raise NotImplementedError


MatchingPart = dict[str, list[str]]
MatchingParts = dict[str, MatchingPart]


class TestBuildMatchingParts:
    def test_query_database_matching_true_has_no_cols_equal(self) -> None:
        parts = _build_matching_parts(["ffl", "ffr"], ["hfl", "hfr"], database_query_matching=True)
        for part in parts.values():
            assert part["cols_equal"] == []

    def test_query_database_matching_false_requires_identity_and_encounter_equal(self) -> None:
        parts = _build_matching_parts(["ffl", "ffr"], ["hfl", "hfr"], database_query_matching=False)
        for part in parts.values():
            assert part["cols_equal"] == ["identity", "encounter_id"]

    def test_labels_are_wired_correctly(self) -> None:
        parts = _build_matching_parts(["front"], ["hind"], database_query_matching=True)
        assert parts["heads"]["labels"] == ["head"]
        assert parts["front flippers"]["labels"] == ["front"]
        assert parts["hind flippers"]["labels"] == ["hind"]
        assert parts["bodies"]["labels"] == ["turtle"]

    def test_flippers_have_cols_unequal_but_heads_and_bodies_do_not(self) -> None:
        parts = _build_matching_parts(["front"], ["hind"], database_query_matching=True)
        assert parts["front flippers"]["cols_unequal"] == ["label", "orientation"]
        assert parts["hind flippers"]["cols_unequal"] == ["label", "orientation"]
        assert parts["heads"]["cols_unequal"] == []
        assert parts["bodies"]["cols_unequal"] == []


class TestCombineMatchingParts:
    def test_union_of_labels(self) -> None:
        parts1: MatchingParts = {"heads": {"labels": ["head"], "cols_equal": [], "cols_unequal": []}}
        parts2: MatchingParts = {"heads": {"labels": ["head2"], "cols_equal": [], "cols_unequal": []}}
        combined = combine_matching_parts(parts1, parts2)
        assert set(combined["heads"]["labels"]) == {"head", "head2"}

    def test_union_of_cols_equal_and_cols_unequal(self) -> None:
        parts1: MatchingParts = {"heads": {"labels": [], "cols_equal": ["identity"], "cols_unequal": ["label"]}}
        parts2: MatchingParts = {"heads": {"labels": [], "cols_equal": ["encounter_id"], "cols_unequal": ["orientation"]}}
        combined = combine_matching_parts(parts1, parts2)
        assert set(combined["heads"]["cols_equal"]) == {"identity", "encounter_id"}
        assert set(combined["heads"]["cols_unequal"]) == {"label", "orientation"}

    def test_duplicate_values_are_deduplicated(self) -> None:
        parts1: MatchingParts = {"heads": {"labels": ["head"], "cols_equal": ["identity"], "cols_unequal": []}}
        parts2: MatchingParts = {"heads": {"labels": ["head"], "cols_equal": ["identity"], "cols_unequal": []}}
        combined = combine_matching_parts(parts1, parts2)
        assert combined["heads"]["labels"] == ["head"]
        assert combined["heads"]["cols_equal"] == ["identity"]

    def test_mismatched_keys_raise_assertion_error(self) -> None:
        parts1: MatchingParts = {"heads": {"labels": [], "cols_equal": [], "cols_unequal": []}}
        parts2: MatchingParts = {"bodies": {"labels": [], "cols_equal": [], "cols_unequal": []}}
        with pytest.raises(AssertionError):
            combine_matching_parts(parts1, parts2)

    def test_all_matching_part_keys_are_preserved(self) -> None:
        parts1: MatchingParts = {
            "heads": {"labels": ["a"], "cols_equal": [], "cols_unequal": []},
            "bodies": {"labels": ["b"], "cols_equal": [], "cols_unequal": []},
        }
        parts2: MatchingParts = {
            "heads": {"labels": ["c"], "cols_equal": [], "cols_unequal": []},
            "bodies": {"labels": ["d"], "cols_equal": [], "cols_unequal": []},
        }
        combined = combine_matching_parts(parts1, parts2)
        assert set(combined.keys()) == {"heads", "bodies"}


class TestCombineDatasets:
    def test_concatenates_metadata_from_all_datasets(self) -> None:
        ds1 = make_wd(pd.DataFrame({"path": ["a.jpg"], "identity": ["x"]}))
        ds2 = make_wd(pd.DataFrame({"path": ["b.jpg"], "identity": ["y"]}))
        combined = combine_datasets([ds1, ds2])
        assert list(combined.metadata["identity"]) == ["x", "y"]

    def test_prepends_root_to_path_when_root_is_set(self, tmp_path: Path) -> None:
        ds1 = make_wd(pd.DataFrame({"path": ["a.jpg"], "identity": ["x"]}), root=str(tmp_path))
        combined = combine_datasets([ds1])
        assert combined.metadata["path"].iloc[0] == f"{tmp_path}/a.jpg"

    def test_does_not_alter_path_when_root_is_none(self) -> None:
        ds1 = make_wd(pd.DataFrame({"path": ["a.jpg"], "identity": ["x"]}), root=None)
        combined = combine_datasets([ds1])
        assert combined.metadata["path"].iloc[0] == "a.jpg"

    def test_prefixes_encounter_id_with_dataset_index_by_default(self) -> None:
        ds1 = make_wd(pd.DataFrame({"path": ["a.jpg"], "identity": ["x"], "encounter_id": ["e1"]}))
        ds2 = make_wd(pd.DataFrame({"path": ["b.jpg"], "identity": ["y"], "encounter_id": ["e1"]}))
        combined = combine_datasets([ds1, ds2])
        assert list(combined.metadata["encounter_id"]) == ["0_e1", "1_e1"]

    def test_custom_encounter_prefixes(self) -> None:
        ds1 = make_wd(pd.DataFrame({"path": ["a.jpg"], "identity": ["x"], "encounter_id": ["e1"]}))
        ds2 = make_wd(pd.DataFrame({"path": ["b.jpg"], "identity": ["y"], "encounter_id": ["e1"]}))
        combined = combine_datasets([ds1, ds2], encounter_prefixes=["new", "master"])
        assert list(combined.metadata["encounter_id"]) == ["new_e1", "master_e1"]

    def test_null_encounter_id_left_untouched(self) -> None:
        ds1 = make_wd(pd.DataFrame({"path": ["a.jpg", "b.jpg"], "identity": ["x", "y"], "encounter_id": ["e1", None]}))
        combined = combine_datasets([ds1])
        assert combined.metadata["encounter_id"].iloc[0] == "0_e1"
        assert pd.isnull(combined.metadata["encounter_id"].iloc[1])

    def test_original_dataset_metadata_not_mutated(self, tmp_path: Path) -> None:
        ds1 = make_wd(pd.DataFrame({"path": ["a.jpg"], "identity": ["x"], "encounter_id": ["e1"]}), root=str(tmp_path))
        original_path = ds1.metadata["path"].iloc[0]
        combine_datasets([ds1])
        assert ds1.metadata["path"].iloc[0] == original_path


class TestMatchingDatasetRemoveKeepUnknown:
    def _make_dataset(self) -> ConcreteFlippersMerged:
        df = pd.DataFrame(
            {
                "path": ["a.jpg", "b.jpg", "c.jpg"],
                "identity": ["x", "unknown", "y"],
            }
        )
        wildlife_dataset = make_wd(df)
        return ConcreteFlippersMerged(wildlife_dataset, database_query_matching=True)

    def test_remove_unknown_drops_unknown_rows(self) -> None:
        matching_dataset = self._make_dataset()
        matching_dataset.remove_unknown()
        assert set(matching_dataset.wildlife_dataset.metadata["identity"]) == {"x", "y"}

    def test_keep_only_unknown_keeps_only_unknown_rows(self) -> None:
        matching_dataset = self._make_dataset()
        matching_dataset.keep_only_unknown()
        assert list(matching_dataset.wildlife_dataset.metadata["identity"]) == ["unknown"]

    def test_from_paths_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            MatchingDataset(make_wd(pd.DataFrame({"path": ["a"], "identity": ["x"]})), {})  # type: ignore[abstract]


class TestTurtleDetectorSegmentedSelectSubset:
    def _dataset(self, rows: list[dict[str, Any]]) -> ConcreteSegmented:
        df = pd.DataFrame(rows)
        wildlife_dataset = make_wd(df)
        return ConcreteSegmented(wildlife_dataset, database_query_matching=True)

    def test_keeps_boxes_above_threshold_when_enough(self) -> None:
        rows = [
            {"path": "1.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.9},
            {"path": "2.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.85},
            {"path": "3.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.5},
        ]
        matching_dataset = self._dataset(rows)
        matching_dataset.select_subset(score_threshold=0.8, score_min=0.3, max_per_encounter=10, min_per_encounter=2)
        kept_scores = sorted(matching_dataset.wildlife_dataset.metadata["score"], reverse=True)
        assert kept_scores == [0.9, 0.85]

    def test_falls_back_to_min_per_encounter_when_not_enough_above_threshold(self) -> None:
        rows = [
            {"path": "1.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.5},
            {"path": "2.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.4},
            {"path": "3.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.35},
            {"path": "4.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.1},
        ]
        matching_dataset = self._dataset(rows)
        matching_dataset.select_subset(score_threshold=0.8, score_min=0.3, max_per_encounter=10, min_per_encounter=2)
        kept_scores = sorted(matching_dataset.wildlife_dataset.metadata["score"], reverse=True)
        assert kept_scores == [0.5, 0.4]

    def test_score_min_filters_out_low_scores_before_min_per_encounter_fallback(self) -> None:
        rows = [
            {"path": "1.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.5},
            {"path": "2.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.1},
            {"path": "3.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.05},
        ]
        matching_dataset = self._dataset(rows)
        matching_dataset.select_subset(score_threshold=0.8, score_min=0.3, max_per_encounter=10, min_per_encounter=2)
        # only one row survives the score_min filter, so fewer than min_per_encounter remain
        kept_scores = list(matching_dataset.wildlife_dataset.metadata["score"])
        assert kept_scores == [0.5]

    def test_max_per_encounter_caps_candidates_before_filtering(self) -> None:
        rows = [{"path": f"{i}.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.9 - i * 0.01} for i in range(5)]
        matching_dataset = self._dataset(rows)
        matching_dataset.select_subset(score_threshold=0.0, score_min=0.0, max_per_encounter=2, min_per_encounter=1)
        assert len(matching_dataset.wildlife_dataset.metadata) == 2

    def test_groups_are_independent_per_label_and_encounter(self) -> None:
        rows = [
            {"path": "1.jpg", "identity": "x", "label": "head", "encounter_id": "e1", "score": 0.9},
            {"path": "2.jpg", "identity": "x", "label": "turtle", "encounter_id": "e1", "score": 0.9},
            {"path": "3.jpg", "identity": "y", "label": "head", "encounter_id": "e2", "score": 0.9},
        ]
        matching_dataset = self._dataset(rows)
        matching_dataset.select_subset(score_threshold=0.8, score_min=0.3, max_per_encounter=10, min_per_encounter=1)
        assert len(matching_dataset.wildlife_dataset.metadata) == 3


class FakeTWE:
    def __init__(
        self,
        root: str | None = None,
        file_name: str | None = None,
        load_segmentation: bool | None = None,
        df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        self.root = root
        self.file_name = file_name
        self.load_segmentation = load_segmentation
        if df is not None:
            self.df = df
        else:
            self.df = pd.DataFrame(
                {
                    "path": ["a.jpg"],
                    "identity": ["x"],
                    "encounter_id": ["e1"],
                }
            )


class TestCombineDatasetsTwe:
    def test_wires_root_paths_and_prefixes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created: list[dict[str, Any]] = []

        class RecordingFakeTWE(FakeTWE):
            def __init__(self, root: str | None = None, file_name: str | None = None, load_segmentation: bool | None = None, df: pd.DataFrame | None = None, **kwargs: Any) -> None:
                created.append({"root": root, "file_name": file_name, "load_segmentation": load_segmentation})
                super().__init__(root=root, file_name=file_name, load_segmentation=load_segmentation, df=df, **kwargs)

        monkeypatch.setattr(datasets_module.wd_datasets, "TurtlewatchEgypt_New", RecordingFakeTWE)

        result = combine_datasets_twe(
            master_root="/master",
            roots=["2019", "2020"],
            file_name="individuals.csv",
            encounter_prefixes=["p19", "p20"],
        )

        assert created[0]["root"] == "/master/2019"
        assert created[1]["root"] == "/master/2020"
        assert list(result.df["path"]) == ["2019/a.jpg", "2020/a.jpg"]
        assert list(result.df["encounter_id"]) == ["p19_e1", "p20_e1"]

    def test_default_encounter_prefixes_are_positional_indices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(datasets_module.wd_datasets, "TurtlewatchEgypt_New", FakeTWE)
        result = combine_datasets_twe(master_root="/m", roots=["r1", "r2"], file_name="f.csv")
        assert list(result.df["encounter_id"]) == ["0_e1", "1_e1"]

    def test_combine_datasets_twe_from_paths_extracts_expected_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_combine(master_root: str, roots: list[str], file_name: str, **kwargs: Any) -> str:
            captured.update(master_root=master_root, roots=roots, file_name=file_name, kwargs=kwargs)
            return "combined_dataset"

        monkeypatch.setattr(datasets_module, "combine_datasets_twe", fake_combine)
        paths = {
            "root_new": "/data/new",
            "roots_new": ["2019", "2020"],
            "file_name_individuals": "/data/new/individuals.csv",
        }
        result = combine_datasets_twe_from_paths(paths)
        assert result == "combined_dataset"
        assert captured["master_root"] == "/data/new"
        assert captured["roots"] == ["2019", "2020"]
        assert captured["file_name"] == "/data/new/individuals.csv"


class TestWrapperClassesWiring:
    def test_turtles_of_smsrc_wires_database_flag_and_calls_select_subset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        df = pd.DataFrame(
            {
                "path": ["a.jpg"],
                "identity": ["x"],
                "label": ["head"],
                "encounter_id": ["e1"],
                "score": [0.9],
            }
        )
        captured_kwargs: dict[str, Any] = {}

        class FakeSMSRC:
            def __init__(self, root: str, img_load: str | None = None, load_segmentation: bool | None = None, load_label: bool | None = None, **kwargs: Any) -> None:
                captured_kwargs.update(root=root, img_load=img_load, load_segmentation=load_segmentation, load_label=load_label)
                self.metadata = df.copy()
                self.root = root

            def get_subset(self, mask: pd.Series) -> "FakeSMSRC":
                fake = FakeSMSRC.__new__(FakeSMSRC)
                fake.metadata = self.metadata.loc[mask].reset_index(drop=True)
                fake.root = self.root
                return fake

        monkeypatch.setattr(datasets_module.wd_datasets, "TurtlesOfSMSRC", FakeSMSRC)

        matching_dataset = TurtlesOfSMSRC("/data/root", is_database=True)

        assert captured_kwargs["root"] == "/data/root"
        assert captured_kwargs["img_load"] == "bbox"
        assert captured_kwargs["load_segmentation"] is True
        assert captured_kwargs["load_label"] is True
        assert matching_dataset.matching_parts["heads"]["cols_equal"] == []

    def test_turtles_of_smsrc_from_paths_uses_root_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: dict[str, Any] = {}

        def fake_init(self: TurtlesOfSMSRC, root: str, is_database: bool, dataset_kwargs: dict[str, Any] | None = None, subset_kwargs: dict[str, Any] | None = None) -> None:
            received.update(root=root, is_database=is_database)

        monkeypatch.setattr(TurtlesOfSMSRC, "__init__", fake_init)
        TurtlesOfSMSRC.from_paths({"root": "/some/root"}, is_database=False)
        assert received == {"root": "/some/root", "is_database": False}

    def test_turtlewatch_egypt_citizen_from_paths_uses_root_citizen_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: dict[str, Any] = {}

        def fake_init(self: TurtlewatchEgypt_Citizen, root: str, is_database: bool, dataset_kwargs: dict[str, Any] | None = None, subset_kwargs: dict[str, Any] | None = None) -> None:
            received.update(root=root, is_database=is_database)

        monkeypatch.setattr(TurtlewatchEgypt_Citizen, "__init__", fake_init)
        TurtlewatchEgypt_Citizen.from_paths({"root_citizen": "/citizen/root"}, is_database=True)
        assert received == {"root": "/citizen/root", "is_database": True}

    def test_turtlewatch_egypt_new_from_paths_uses_new_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: dict[str, Any] = {}

        def fake_init(
            self: TurtlewatchEgypt_New,
            root_master: str,
            roots: list[str],
            is_database: bool,
            file_name: str,
            dataset_kwargs: dict[str, Any] | None = None,
            subset_kwargs: dict[str, Any] | None = None,
        ) -> None:
            received.update(root_master=root_master, roots=roots, is_database=is_database, file_name=file_name)

        monkeypatch.setattr(TurtlewatchEgypt_New, "__init__", fake_init)
        paths = {
            "root_new": "/new/root",
            "roots_new": ["2019", "2020"],
            "file_name_individuals": "/new/root/individuals.csv",
        }
        TurtlewatchEgypt_New.from_paths(paths, is_database=False)
        assert received["root_master"] == "/new/root"
        assert received["roots"] == ["2019", "2020"]
        assert received["is_database"] is False
        assert received["file_name"] == "/new/root/individuals.csv"

    def test_turtlewatch_egypt_master_from_paths_uses_flipper_and_head_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: dict[str, Any] = {}

        def fake_init(
            self: TurtlewatchEgypt_Master,
            root_heads: str,
            root_flippers_front: str,
            root_flippers_hind: str,
            database_query_matching: bool,
            file_name: str,
            dataset_kwargs: dict[str, Any] | None = None,
            score_min: float | None = None,
        ) -> None:
            received.update(
                root_heads=root_heads,
                root_flippers_front=root_flippers_front,
                root_flippers_hind=root_flippers_hind,
                database_query_matching=database_query_matching,
                file_name=file_name,
            )

        monkeypatch.setattr(TurtlewatchEgypt_Master, "__init__", fake_init)
        paths = {
            "root_heads": "/h",
            "root_flippers_front": "/ff",
            "root_flippers_hind": "/fh",
            "file_name_individuals": "/individuals.csv",
        }
        TurtlewatchEgypt_Master.from_paths(paths, database_query_matching=True)
        assert received["root_heads"] == "/h"
        assert received["root_flippers_front"] == "/ff"
        assert received["root_flippers_hind"] == "/fh"
        assert received["database_query_matching"] is True
        assert received["file_name"] == "/individuals.csv"

    def test_sea_turtle_id_2022_from_paths_uses_root_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: dict[str, Any] = {}

        def fake_init(self: Any, root: str, database_query_matching: bool, dataset_kwargs: dict[str, Any] | None = None) -> None:
            received.update(root=root, database_query_matching=database_query_matching)

        monkeypatch.setattr(datasets_module.SeaTurtleID2022, "__init__", fake_init)
        datasets_module.SeaTurtleID2022.from_paths({"root": "/sea/root"}, database_query_matching=False)
        assert received == {"root": "/sea/root", "database_query_matching": False}

    def test_turtlewatch_egypt_combined_from_paths_uses_all_new_and_master_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: dict[str, Any] = {}

        def fake_init(
            self: TurtlewatchEgypt_Combined,
            root_master: str,
            roots: list[str],
            root_heads: str,
            root_flippers_front: str,
            root_flippers_hind: str,
            file_name: str,
            dataset_kwargs: dict[str, Any] | None = None,
            subset_kwargs: dict[str, Any] | None = None,
        ) -> None:
            received.update(
                root_master=root_master,
                roots=roots,
                root_heads=root_heads,
                root_flippers_front=root_flippers_front,
                root_flippers_hind=root_flippers_hind,
                file_name=file_name,
            )

        monkeypatch.setattr(TurtlewatchEgypt_Combined, "__init__", fake_init)
        paths = {
            "root_new": "/new",
            "roots_new": ["2019"],
            "root_heads": "/heads",
            "root_flippers_front": "/ff",
            "root_flippers_hind": "/fh",
            "file_name_individuals": "/new/individuals.csv",
        }
        TurtlewatchEgypt_Combined.from_paths(paths)
        assert received["root_master"] == "/new"
        assert received["roots"] == ["2019"]
        assert received["root_heads"] == "/heads"
        assert received["root_flippers_front"] == "/ff"
        assert received["root_flippers_hind"] == "/fh"
        assert received["file_name"] == "/new/individuals.csv"
