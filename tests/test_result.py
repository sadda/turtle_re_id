from typing import Any

import numpy as np

from animal_re_id.similarity.result import SimilarityResult


def _make_result(**overrides: Any) -> SimilarityResult:
    defaults: dict[str, Any] = dict(
        similarity=np.array([[1.0, 0.5], [0.2, 0.9]]),
        query="query_dataset",
        database="database_dataset",
        name="dino",
        B=5,
        img_load="bbox",
        matching_part="heads",
        match_opposite_sides="allow_all",
        split_year="2021",
    )
    defaults.update(overrides)
    return SimilarityResult(**defaults)


class TestSimilarityResult:
    def test_to_dict_contains_all_fields(self) -> None:
        result = _make_result()
        d = result.to_dict()
        assert set(d.keys()) == {
            "similarity",
            "query",
            "database",
            "name",
            "B",
            "img_load",
            "matching_part",
            "match_opposite_sides",
            "split_year",
        }

    def test_to_dict_preserves_values(self) -> None:
        result = _make_result(name="mega", B=None)
        d = result.to_dict()
        assert d["name"] == "mega"
        assert d["B"] is None
        assert np.array_equal(d["similarity"], result.similarity)

    def test_to_dict_is_a_plain_dict(self) -> None:
        result = _make_result()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert not isinstance(d, SimilarityResult)
