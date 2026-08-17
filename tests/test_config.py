import json
import os
from pathlib import Path

import pytest

from animal_re_id.config import DEFAULT_PATHS_FILE, _deep_merge, load_paths, resolve_path


class TestDeepMerge:
    def test_disjoint_keys_are_combined(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_scalar_override_replaces_value(self) -> None:
        base = {"a": 1}
        override = {"a": 2}
        assert _deep_merge(base, override) == {"a": 2}

    def test_nested_dicts_are_merged_recursively(self) -> None:
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        assert _deep_merge(base, override) == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_dict_overriding_non_dict_replaces_it(self) -> None:
        base = {"a": 1}
        override = {"a": {"x": 1}}
        assert _deep_merge(base, override) == {"a": {"x": 1}}

    def test_non_dict_overriding_dict_replaces_it(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": 1}
        assert _deep_merge(base, override) == {"a": 1}

    def test_base_is_not_mutated(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}

    def test_empty_override_returns_equivalent_base(self) -> None:
        base = {"a": 1, "b": {"c": 2}}
        assert _deep_merge(base, {}) == base


class TestResolvePath:
    def test_expands_user_home(self) -> None:
        resolved = resolve_path("~")
        assert "~" not in resolved

    def test_is_absolute(self) -> None:
        resolved = resolve_path("some/relative/path")
        assert os.path.isabs(resolved)

    def test_resolves_relative_to_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        resolved = resolve_path("foo")
        assert resolved == str(tmp_path / "foo")


class TestLoadPaths:
    def test_default_paths_file_exists(self) -> None:
        assert DEFAULT_PATHS_FILE.exists()

    def test_load_known_dataset_from_defaults(self) -> None:
        paths = load_paths("SeaTurtleID2022")
        assert "root" in paths

    def test_unknown_dataset_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            load_paths("NotARealDataset")

    def test_local_path_overrides_defaults(self, tmp_path: Path) -> None:
        local_file = tmp_path / "local_paths.json"
        local_file.write_text(json.dumps({"SeaTurtleID2022": {"root": "/custom/root"}}))
        paths = load_paths("SeaTurtleID2022", local_path=local_file)
        assert paths["root"] == "/custom/root"
        # other default keys for that dataset should be preserved
        with open(DEFAULT_PATHS_FILE) as f:
            defaults = json.load(f)
        for key in defaults["SeaTurtleID2022"]:
            if key != "root":
                assert paths[key] == defaults["SeaTurtleID2022"][key]

    def test_nonexistent_local_path_is_ignored(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.json"
        paths = load_paths("SeaTurtleID2022", local_path=missing)
        default_paths = load_paths("SeaTurtleID2022")
        assert paths == default_paths

    def test_local_path_can_add_new_dataset(self, tmp_path: Path) -> None:
        local_file = tmp_path / "local_paths.json"
        local_file.write_text(json.dumps({"NewDataset": {"root": "/x"}}))
        paths = load_paths("NewDataset", local_path=local_file)
        assert paths == {"root": "/x"}

    def test_local_path_as_string(self, tmp_path: Path) -> None:
        local_file = tmp_path / "local_paths.json"
        local_file.write_text(json.dumps({"SeaTurtleID2022": {"root": "/from/string/path"}}))
        paths = load_paths("SeaTurtleID2022", local_path=str(local_file))
        assert paths["root"] == "/from/string/path"
