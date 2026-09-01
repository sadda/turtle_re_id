from pathlib import Path

import pytest

from animal_re_id.similarity.cache import SimilarityCache


@pytest.fixture
def cache(tmp_path: Path) -> SimilarityCache:
    return SimilarityCache(str(tmp_path))


class TestPath:
    def test_joins_root_and_joins_parts_with_underscore(self, cache: SimilarityCache) -> None:
        path = cache.path(["dino", 5, "bbox", "heads", "allow_all", "2021"])
        assert path == cache.root + "/dino_5_bbox_heads_allow_all_2021.pkl"

    def test_non_string_parts_are_stringified(self, cache: SimilarityCache) -> None:
        path = cache.path(["a", None, 1, True])
        assert path.endswith("a_None_1_True.pkl")


class TestExistsSaveLoad:
    def test_exists_false_when_not_saved(self, cache: SimilarityCache) -> None:
        assert cache.exists(["missing"]) is False

    def test_save_then_exists_true(self, cache: SimilarityCache) -> None:
        cache.save(["k1"], {"a": 1})
        assert cache.exists(["k1"]) is True

    def test_save_then_load_roundtrip(self, cache: SimilarityCache) -> None:
        data = {"similarity": [[1, 2], [3, 4]], "name": "dino"}
        cache.save(["k2"], data)
        loaded = cache.load(["k2"])
        assert loaded == data

    def test_load_missing_raises(self, cache: SimilarityCache) -> None:
        with pytest.raises(FileNotFoundError):
            cache.load(["missing"])

    def test_save_overwrites_existing(self, cache: SimilarityCache) -> None:
        cache.save(["k3"], {"v": 1})
        cache.save(["k3"], {"v": 2})
        assert cache.load(["k3"]) == {"v": 2}

    def test_different_key_parts_map_to_different_files(self, cache: SimilarityCache) -> None:
        cache.save(["a", 1], {"v": "first"})
        cache.save(["a", 2], {"v": "second"})
        assert cache.load(["a", 1]) == {"v": "first"}
        assert cache.load(["a", 2]) == {"v": "second"}

    def test_save_creates_root_directory_if_missing(self, tmp_path: Path) -> None:
        cache = SimilarityCache(str(tmp_path / "nested" / "dir"))
        cache.save(["k"], {"v": 1})
        assert cache.load(["k"]) == {"v": 1}
