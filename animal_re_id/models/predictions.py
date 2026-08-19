from copy import deepcopy

import numpy as np
import pandas as pd
from wildlife_tools.inference import TopkClassifier

DEF_VALUES = [-np.inf, "", "", "", -1, -1]


class EncounterPredictions:
    def __init__(self, similarity, query, database, encounter_col: str = "encounter_id", identity_col: str = "identity") -> None:
        self.similarity = similarity
        self.query = query
        self.database = database
        self.encounter_col = encounter_col
        self.identity_col = identity_col

    def group_by_encounter(self) -> None:
        encounter_groups = self.query.metadata.groupby(self.encounter_col).indices
        self.encounter_ids = list(encounter_groups.keys())
        self.identities_database = self.database.metadata[self.identity_col].to_numpy()

        # TODO: this recomputes it every time
        similarity_by_encounter, query_iloc_by_encounter = [], []
        for row_idx in encounter_groups.values():
            sub = self.similarity[row_idx]
            argmax = sub.argmax(axis=0)
            similarity_by_encounter.append(sub[argmax, np.arange(sub.shape[1])])
            query_iloc_by_encounter.append(row_idx[argmax])
        self.similarity_by_encounter = np.vstack(similarity_by_encounter)
        self.query_iloc_by_encounter = np.vstack(query_iloc_by_encounter)

    def classify(self, matching_part: str, matcher: str, k_max: int | None = None) -> None:
        if k_max is None:
            k_max = len(self.database)
        classifier = TopkClassifier(self.identities_database, k=k_max, return_all=True)
        predictions, scores, database_iloc = classifier(self.similarity_by_encounter)
        scores = scores.astype(np.float64)
        scores_dict = dict(zip(self.encounter_ids, scores))
        predictions_dict = dict(zip(self.encounter_ids, predictions))
        database_iloc_dict = dict(zip(self.encounter_ids, database_iloc))
        query_iloc_dict = {
            encounter_id: self.query_iloc_by_encounter[i][database_iloc[i]] for i, encounter_id in enumerate(self.encounter_ids)
        }
        self._set_predictions(scores_dict, predictions_dict, matching_part, matcher, database_iloc_dict, query_iloc_dict, k_max)

    def _set_predictions(
        self,
        scores: dict,
        predictions: dict,
        matching_parts: str | dict,
        matchers: str | dict,
        database_iloc: dict,
        query_iloc: dict,
        k_max: int,
    ) -> None:
        self.scores = scores
        self.predictions = predictions
        self.database_iloc = database_iloc
        self.query_iloc = query_iloc
        self.k_max = k_max

        self.matching_parts = self._resolve_tag(matching_parts)
        self.matchers = self._resolve_tag(matchers)
        self._extend_data(k_max)

    def _extend_data(self, k_max):
        data = [self.scores, self.predictions, self.matching_parts, self.matchers, self.database_iloc, self.query_iloc]
        for var, fill_value in zip(data, DEF_VALUES):
            for encounter, value in var.items():
                var[encounter] = array_extend(value, k_max, fill_value)

    def _resolve_tag(self, tag: str | dict) -> dict:
        if isinstance(tag, str):
            return {encounter: np.full(len(score), tag) for encounter, score in self.scores.items()}
        elif isinstance(tag, dict):
            return tag
        else:
            raise ValueError("tag must be str or dict")

    def compute_matches(self, k: int, identities: dict[str, str]) -> int:
        if k == 1:
            return np.sum([identities[encounter] == pred[0] for encounter, pred in self.predictions.items()])
        else:
            return np.sum([np.isin(identities[encounter], pred[:k]) for encounter, pred in self.predictions.items()])

    def compute_matches_by_part(self, k: int, identities: dict[str, str]) -> list[tuple]:
        if k == 1:
            results = []
            for encounter, pred in self.predictions.items():
                success = identities[encounter] == pred[0]
                score = self.scores[encounter][0]
                matching_part = self.matching_parts[encounter][0]
                results.append((success, score, matching_part))
        else:
            raise NotImplementedError("Not yet implemented")
        return results

    def extract_highest_predictions(
        self,
        max_matches: int,
        max_per_identity: int,
        unlimited_unknown: bool = True,
        min_similarity: float = 0,
    ) -> pd.DataFrame:
        # Flatten scores/predictions/matching_parts/matchers/database_iloc/query_iloc (dict[encounter -> array])
        # into one row per (encounter, candidate identity).
        encounters, scores, identities2, matching_part, matcher, database_iloc, query_iloc = [], [], [], [], [], [], []
        for encounter in self.scores:
            n = len(self.scores[encounter])
            encounters.extend([encounter] * n)
            scores.extend(self.scores[encounter])
            identities2.extend(self.predictions[encounter])
            matching_part.extend(self.matching_parts[encounter])
            matcher.extend(self.matchers[encounter])
            database_iloc.extend(self.database_iloc[encounter])
            query_iloc.extend(self.query_iloc[encounter])

        results = pd.DataFrame(
            {
                "encounter1": encounters,
                "identity2": identities2,
                "matching_part": matching_part,
                "matcher": matcher,
                "similarity_orig": scores,
                "database_iloc": database_iloc,
                "query_iloc": query_iloc,
            }
        )

        # Normalize similarity per (matching_part, matcher) (matching the old gather_df convention).
        # Sorting/filtering/capping below all use this normalized value, exactly as the old code did.
        results["similarity"] = results.groupby(["matching_part", "matcher"])["similarity_orig"].transform(lambda s: s / s.max())

        results = results[results["similarity"] > min_similarity]
        results = results.sort_values("similarity", ascending=False).reset_index(drop=True)

        results["identity1"] = self.query.metadata["identity"].to_numpy()[results["query_iloc"]]
        results["encounter2"] = self.database.metadata["encounter_id"].to_numpy()[results["database_iloc"]]
        results["image_id1"] = self.query.metadata["image_id"].to_numpy()[results["query_iloc"]]
        results["image_id2"] = self.database.metadata["image_id"].to_numpy()[results["database_iloc"]]

        # Canonicalize pair order (image_id1 < image_id2) so an asymmetric similarity's two
        # directions of the same physical pair present consistently rather than as separate rows.
        swap = results["image_id1"] > results["image_id2"]
        sided_cols = [col[:-1] for col in results.columns if col.endswith("1") and f"{col[:-1]}2" in results.columns]
        for col in sided_cols:
            col1, col2 = f"{col}1", f"{col}2"
            results.loc[swap, [col1, col2]] = results.loc[swap, [col2, col1]].to_numpy()

        # Keep only one match between unique images (the two directions of a pair now collide)
        results = results.drop_duplicates(subset=["image_id1", "image_id2", "matching_part"]).reset_index(drop=True)

        # Keep only at most max_per_identity matches per a pair of identities (unordered)
        identity_pair = results[["identity1", "identity2"]]
        identity_lo, identity_hi = identity_pair.min(axis=1), identity_pair.max(axis=1)
        idx_remove = []
        for (lo, hi), group in results.groupby([identity_lo, identity_hi], sort=False):
            if unlimited_unknown and "unknown" in (lo, hi):
                continue
            idx_remove.extend(group.index[max_per_identity:])
        results = results.drop(index=idx_remove)

        # Keep only at most max_matches matches overall
        results = results.iloc[:max_matches]

        columns = [
            "encounter1",
            "encounter2",
            "identity1",
            "identity2",
            "image_id1",
            "image_id2",
            "matching_part",
            "matcher",
            "similarity",
            "similarity_orig",
        ]
        return results[columns].reset_index(drop=True)


def array_extend(x, n, fill_value):
    n_missing = n - len(x)
    if n_missing > 0:
        return np.concatenate((x, np.full(n_missing, fill_value)))
    return x


def _offset_iloc(iloc_dict: dict, offset: int) -> dict:
    if offset == 0:
        return iloc_dict
    return {encounter: np.where(value == -1, -1, value + offset) for encounter, value in iloc_dict.items()}


def _merge_wildlife_datasets(datasets: list):
    first = datasets[0]
    for dataset in datasets:
        if dataset.root != first.root or dataset.img_load != first.img_load:
            raise ValueError("All EncounterPredictions must share the same dataset root and img_load to be merged")

    merged = deepcopy(first)
    merged.df = pd.concat([dataset.metadata for dataset in datasets], ignore_index=True)
    return merged


def merge_data(data: list[EncounterPredictions]) -> EncounterPredictions:
    if not hasattr(data[0], "k_max"):
        raise ValueError("All EncounterPredictions must be classified (via classify()) before merging")
    k_max = data[0].k_max

    for d in data:
        if not hasattr(d, "k_max"):
            raise ValueError("All EncounterPredictions must be classified (via classify()) before merging")
        if d.k_max != k_max:
            raise ValueError(f"All EncounterPredictions must already be padded/truncated to k_max={k_max}, got {d.k_max}")

    # Different matching_parts use disjoint row subsets of the underlying dataset (see
    # MatchingDataset.for_matching_part), so query/database - and the iloc positions recorded
    # against them - can legitimately differ between instances. When they do, combine them into
    # one dataset and shift each instance's ilocs into that combined dataset's index space.
    same_query = all(d.query.metadata.equals(data[0].query.metadata) for d in data)
    if same_query:
        merged_query = data[0].query
        query_offsets = [0] * len(data)
    else:
        merged_query = _merge_wildlife_datasets([d.query for d in data])
        query_offsets = np.cumsum([0] + [len(d.query.metadata) for d in data])[:-1]

    same_database = all(d.database.metadata.equals(data[0].database.metadata) for d in data)
    if same_database:
        merged_database = data[0].database
        database_offsets = [0] * len(data)
    else:
        merged_database = _merge_wildlife_datasets([d.database for d in data])
        database_offsets = np.cumsum([0] + [len(d.database.metadata) for d in data])[:-1]

    # Each instance already carries its own source label per encounter via .matching_parts/.matchers
    # (set by classify()), so both are merged the same way as scores/predictions/etc below.
    scores = [d.scores for d in data]
    predictions = [d.predictions for d in data]
    matching_parts = [d.matching_parts for d in data]
    matchers = [d.matchers for d in data]
    database_iloc = [_offset_iloc(d.database_iloc, database_offsets[i]) for i, d in enumerate(data)]
    query_iloc = [_offset_iloc(d.query_iloc, query_offsets[i]) for i, d in enumerate(data)]
    encounters = set().union(*scores)

    # For each encounter, combine candidates from every source, then sort/dedupe/truncate to k_max
    fields = [scores, predictions, matching_parts, matchers, database_iloc, query_iloc]
    merged = [{} for _ in fields]
    for encounter in encounters:
        arrays = [
            np.concatenate([source.get(encounter, np.full(k_max, fill_value)) for source in field])
            for field, fill_value in zip(fields, DEF_VALUES)
        ]

        # Sort the values starting with highest scores
        idx = np.argsort(arrays[0])[::-1]
        arrays = [a[idx] for a in arrays]

        # Keep only unique predictions
        _, idx = np.unique(arrays[1], return_index=True)
        arrays = [a[idx] for a in arrays]

        # Sort again by score, since np.unique reorders by value rather than by score
        idx = np.argsort(arrays[0])[::-1]
        arrays = [a[idx] for a in arrays]

        # Extract the highest scores
        for merged_field, array, fill_value in zip(merged, arrays, DEF_VALUES):
            merged_field[encounter] = array_extend(array, k_max, fill_value)[:k_max]
    scores_merged, predictions_merged, matching_parts_merged, matchers_merged, database_iloc_merged, query_iloc_merged = merged

    result = EncounterPredictions.__new__(EncounterPredictions)
    result.query = merged_query
    result.database = merged_database
    result._set_predictions(scores_merged, predictions_merged, matching_parts_merged, matchers_merged, database_iloc_merged, query_iloc_merged, k_max)
    return result
