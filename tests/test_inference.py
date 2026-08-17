from typing import Any

import numpy as np
import pandas as pd
import pytest

from animal_re_id.models.inference import best_matches_discovery, best_matches_matching, top_indices


def make_results(rows: Any) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestBestMatchesMatching:
    def test_raises_on_nan_encounter1(self) -> None:
        results = make_results(
            [
                {"encounter1": None, "image_id1": 1, "image_id2": 2, "similarity": 0.9},
            ]
        )
        with pytest.raises(ValueError, match="nan in encounter1"):
            best_matches_matching(results, max_per_encounter=1)

    def test_filters_below_min_similarity(self) -> None:
        results = make_results(
            [
                {"encounter1": "e1", "image_id1": 1, "image_id2": 2, "similarity": 0.05},
                {"encounter1": "e1", "image_id1": 1, "image_id2": 3, "similarity": 0.5},
            ]
        )
        out = best_matches_matching(results, max_per_encounter=10, min_similarity=0.1)
        assert len(out) == 1
        assert out.iloc[0]["image_id2"] == 3

    def test_sorted_descending_by_similarity(self) -> None:
        results = make_results(
            [
                {"encounter1": "e1", "image_id1": 1, "image_id2": 2, "similarity": 0.2},
                {"encounter1": "e1", "image_id1": 1, "image_id2": 3, "similarity": 0.8},
            ]
        )
        out = best_matches_matching(results, max_per_encounter=10)
        assert list(out["similarity"]) == [0.8, 0.2]

    def test_deduplicates_image_id_pairs(self) -> None:
        results = make_results(
            [
                {"encounter1": "e1", "image_id1": 1, "image_id2": 2, "similarity": 0.9},
                {"encounter1": "e1", "image_id1": 1, "image_id2": 2, "similarity": 0.1},
            ]
        )
        out = best_matches_matching(results, max_per_encounter=10)
        assert len(out) == 1
        assert out.iloc[0]["similarity"] == 0.9

    def test_limits_matches_per_encounter(self) -> None:
        rows = [{"encounter1": "e1", "image_id1": i, "image_id2": i + 100, "similarity": 1.0 - i * 0.01} for i in range(5)]
        results = make_results(rows)
        out = best_matches_matching(results, max_per_encounter=2)
        assert len(out) == 2
        assert list(out["similarity"]) == pytest.approx([1.0, 0.99])

    def test_max_per_encounter_applies_independently_per_encounter(self) -> None:
        rows = [
            {"encounter1": "e1", "image_id1": 1, "image_id2": 10, "similarity": 0.9},
            {"encounter1": "e1", "image_id1": 2, "image_id2": 11, "similarity": 0.8},
            {"encounter1": "e2", "image_id1": 3, "image_id2": 12, "similarity": 0.7},
        ]
        results = make_results(rows)
        out = best_matches_matching(results, max_per_encounter=1)
        assert len(out) == 2
        assert set(out["encounter1"]) == {"e1", "e2"}

    def test_default_min_similarity_excludes_nonpositive(self) -> None:
        results = make_results(
            [
                {"encounter1": "e1", "image_id1": 1, "image_id2": 2, "similarity": 0.0},
                {"encounter1": "e1", "image_id1": 1, "image_id2": 3, "similarity": -0.1},
                {"encounter1": "e1", "image_id1": 1, "image_id2": 4, "similarity": 0.1},
            ]
        )
        out = best_matches_matching(results, max_per_encounter=10)
        assert len(out) == 1
        assert out.iloc[0]["image_id2"] == 4

    def test_empty_input_returns_empty(self) -> None:
        results = make_results({"encounter1": [], "image_id1": [], "image_id2": [], "similarity": []})
        out = best_matches_matching(results, max_per_encounter=10)
        assert len(out) == 0

    def test_index_is_reset(self) -> None:
        rows = [{"encounter1": "e1", "image_id1": i, "image_id2": i + 100, "similarity": 1.0 - i * 0.01} for i in range(3)]
        results = make_results(rows)
        out = best_matches_matching(results, max_per_encounter=10)
        assert list(out.index) == list(range(len(out)))


class TestBestMatchesDiscovery:
    def test_filters_below_min_similarity(self) -> None:
        results = make_results(
            [
                {"identity1": "a", "identity2": "b", "image_id1": 1, "image_id2": 2, "similarity": 0.05},
                {"identity1": "a", "identity2": "c", "image_id1": 1, "image_id2": 3, "similarity": 0.5},
            ]
        )
        out = best_matches_discovery(results, max_matches=10, max_per_identity=10, min_similarity=0.1)
        assert len(out) == 1
        assert out.iloc[0]["identity2"] == "c"

    def test_limits_max_matches(self) -> None:
        rows = [{"identity1": "a", "identity2": f"id{i}", "image_id1": i, "image_id2": i + 100, "similarity": 1.0 - i * 0.01} for i in range(5)]
        results = make_results(rows)
        out = best_matches_discovery(results, max_matches=2, max_per_identity=10)
        assert len(out) == 2

    def test_limits_max_per_identity_pair(self) -> None:
        rows = [
            {"identity1": "a", "identity2": "b", "image_id1": 1, "image_id2": 10, "similarity": 0.9},
            {"identity1": "a", "identity2": "b", "image_id1": 2, "image_id2": 11, "similarity": 0.8},
            {"identity1": "b", "identity2": "a", "image_id1": 3, "image_id2": 12, "similarity": 0.7},
        ]
        results = make_results(rows)
        out = best_matches_discovery(results, max_matches=10, max_per_identity=1)
        # identity pair {a,b} treated as same "key" regardless of ordering -> only 1 kept
        assert len(out) == 1
        assert out.iloc[0]["similarity"] == 0.9

    def test_unlimited_unknown_bypasses_max_per_identity(self) -> None:
        rows = [
            {"identity1": "a", "identity2": "unknown", "image_id1": 1, "image_id2": 10, "similarity": 0.9},
            {"identity1": "a", "identity2": "unknown", "image_id1": 2, "image_id2": 11, "similarity": 0.8},
            {"identity1": "a", "identity2": "unknown", "image_id1": 3, "image_id2": 12, "similarity": 0.7},
        ]
        results = make_results(rows)
        out = best_matches_discovery(results, max_matches=10, max_per_identity=1, unlimited_unknown=True)
        assert len(out) == 3

    def test_limited_unknown_when_flag_false(self) -> None:
        rows = [
            {"identity1": "a", "identity2": "unknown", "image_id1": 1, "image_id2": 10, "similarity": 0.9},
            {"identity1": "a", "identity2": "unknown", "image_id1": 2, "image_id2": 11, "similarity": 0.8},
        ]
        results = make_results(rows)
        out = best_matches_discovery(results, max_matches=10, max_per_identity=1, unlimited_unknown=False)
        assert len(out) == 1
        assert out.iloc[0]["similarity"] == 0.9

    def test_deduplicates_image_id_pairs(self) -> None:
        rows = [
            {"identity1": "a", "identity2": "b", "image_id1": 1, "image_id2": 2, "similarity": 0.9},
            {"identity1": "a", "identity2": "b", "image_id1": 1, "image_id2": 2, "similarity": 0.1},
        ]
        results = make_results(rows)
        out = best_matches_discovery(results, max_matches=10, max_per_identity=10)
        assert len(out) == 1
        assert out.iloc[0]["similarity"] == 0.9

    def test_index_is_reset(self) -> None:
        rows = [{"identity1": "a", "identity2": f"id{i}", "image_id1": i, "image_id2": i + 100, "similarity": 1.0 - i * 0.01} for i in range(3)]
        results = make_results(rows)
        out = best_matches_discovery(results, max_matches=10, max_per_identity=10)
        assert list(out.index) == list(range(len(out)))


class TestTopIndices:
    def test_returns_descending_order_of_flat_values(self) -> None:
        similarity = np.array([[1.0, 3.0], [2.0, 0.5]])
        rows, cols = top_indices(similarity)
        ordered_values = similarity[rows, cols]
        assert list(ordered_values) == sorted(similarity.flatten(), reverse=True)

    def test_top_index_matches_max_value_location(self) -> None:
        similarity = np.array([[1.0, 3.0], [2.0, 0.5]])
        rows, cols = top_indices(similarity)
        assert (rows[0], cols[0]) == (0, 1)

    def test_ignore_lower_diagonal_excludes_lower_triangle(self) -> None:
        original = np.array(
            [
                [0.1, 0.9, 0.8],
                [0.7, 0.2, 0.6],
                [0.5, 0.4, 0.3],
            ]
        )
        working_copy = original.copy()
        rows, cols = top_indices(working_copy, ignore_lower_diagonal=True)
        # tril_indices_from zeroes out the lower triangle *and* the diagonal,
        # so only the strict upper triangle (row < col) remains finite.
        finite_mask = working_copy[rows, cols] != -np.inf
        assert all(r < c for r, c in zip(rows[finite_mask], cols[finite_mask]))
        assert finite_mask.sum() == 3

    def test_returns_all_positions(self) -> None:
        similarity = np.array([[1.0, 2.0], [3.0, 4.0]])
        rows, cols = top_indices(similarity)
        assert len(rows) == 4
        assert len(cols) == 4
