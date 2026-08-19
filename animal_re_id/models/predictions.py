import numpy as np
import pandas as pd
from wildlife_tools.inference import TopkClassifier

DEF_VALUES = [-np.inf, "", "", -1, -1]


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

    def classify(self, k_max: int, matching_part: str) -> None:
        classifier = TopkClassifier(self.identities_database, k=k_max, return_all=True)
        predictions, scores, database_iloc = classifier(self.similarity_by_encounter)
        scores_dict = dict(zip(self.encounter_ids, scores))
        predictions_dict = dict(zip(self.encounter_ids, predictions))
        database_iloc_dict = dict(zip(self.encounter_ids, database_iloc))
        query_iloc_dict = {
            encounter_id: self.query_iloc_by_encounter[i][database_iloc[i]] for i, encounter_id in enumerate(self.encounter_ids)
        }
        self._set_predictions(scores_dict, predictions_dict, matching_part, database_iloc_dict, query_iloc_dict, k_max)

    def _set_predictions(
        self,
        scores: dict,
        predictions: dict,
        matching_parts: str | dict,
        database_iloc: dict,
        query_iloc: dict,
        k_max=None,
    ) -> None:
        self.scores = scores
        self.predictions = predictions
        self.database_iloc = database_iloc
        self.query_iloc = query_iloc

        self._save_matching_part(matching_parts)
        if k_max is not None:
            self._extend_data(k_max)

    def _extend_data(self, k_max):
        data = [self.scores, self.predictions, self.matching_parts, self.database_iloc, self.query_iloc]
        for var, fill_value in zip(data, DEF_VALUES):
            for encounter, value in var.items():
                var[encounter] = array_extend(value, k_max, fill_value)

    def _save_matching_part(self, matching_parts):
        if isinstance(matching_parts, str):
            self.matching_parts = {}
            for encounter, score in self.scores.items():
                self.matching_parts[encounter] = np.full(len(score), matching_parts)
        elif isinstance(matching_parts, dict):
            self.matching_parts = matching_parts
        else:
            raise ValueError("matching_part must be str or dict")

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
        # Flatten scores/predictions/matching_parts/database_iloc/query_iloc (dict[encounter -> array])
        # into one row per (encounter, candidate identity).
        encounters, scores, identities2, matching_parts, database_iloc, query_iloc = [], [], [], [], [], []
        for encounter in self.scores:
            n = len(self.scores[encounter])
            encounters.extend([encounter] * n)
            scores.extend(self.scores[encounter])
            identities2.extend(self.predictions[encounter])
            matching_parts.extend(self.matching_parts[encounter])
            database_iloc.extend(self.database_iloc[encounter])
            query_iloc.extend(self.query_iloc[encounter])

        results = pd.DataFrame(
            {
                "encounter1": encounters,
                "identity2": identities2,
                "matcher": matching_parts,
                "similarity": scores,
                "database_iloc": database_iloc,
                "query_iloc": query_iloc,
            }
        )
        results = results[results["similarity"] > min_similarity]
        results = results.sort_values("similarity", ascending=False).reset_index(drop=True)

        results["identity1"] = self.query.metadata["identity"].to_numpy()[results["query_iloc"]]
        results["encounter2"] = self.database.metadata["encounter_id"].to_numpy()[results["database_iloc"]]
        results["image_id1"] = self.query.metadata["image_id"].to_numpy()[results["query_iloc"]]
        results["image_id2"] = self.database.metadata["image_id"].to_numpy()[results["database_iloc"]]

        # Keep only at most max_per_identity matches per a pair of identities
        pairs = results.apply(lambda r: frozenset((r["identity1"], r["identity2"])), axis=1)
        idx_remove = []
        for pair, group in results.groupby(pairs, sort=False):
            if unlimited_unknown and "unknown" in pair:
                continue
            idx_remove.extend(group.index[max_per_identity:])
        results = results.drop(index=idx_remove)

        # Keep only at most max_matches matches overall
        results = results.iloc[:max_matches]
        return results.drop(columns=["database_iloc", "query_iloc"]).reset_index(drop=True)


def array_extend(x, n, fill_value):
    n_missing = n - len(x)
    if n_missing > 0:
        return np.concatenate((x, np.full(n_missing, fill_value)))
    return x


def merge_data(data: list[EncounterPredictions], k_max) -> EncounterPredictions:
    # Each instance already carries its own source label per encounter via .matching_parts
    # (set by classify()), so it's merged the same way as scores/predictions/etc below.
    scores = [d.scores for d in data]
    predictions = [d.predictions for d in data]
    matching_parts = [d.matching_parts for d in data]
    database_iloc = [d.database_iloc for d in data]
    query_iloc = [d.query_iloc for d in data]
    encounters = set().union(*scores)

    # For each encounter, combine candidates from every source, then sort/dedupe/truncate to k_max
    scores_merged = {}
    predictions_merged = {}
    sources_merged = {}
    database_iloc_merged = {}
    query_iloc_merged = {}
    for encounter in encounters:
        ss = np.vstack([s.get(encounter, np.full(k_max, DEF_VALUES[0])) for s in scores]).flatten()
        pp = np.vstack([s.get(encounter, np.full(k_max, DEF_VALUES[1])) for s in predictions]).flatten()
        qq = np.vstack([s.get(encounter, np.full(k_max, DEF_VALUES[2])) for s in matching_parts]).flatten()
        rr = np.vstack([s.get(encounter, np.full(k_max, DEF_VALUES[3])) for s in database_iloc]).flatten()
        tt = np.vstack([s.get(encounter, np.full(k_max, DEF_VALUES[4])) for s in query_iloc]).flatten()

        # Sort the values starting with highest scores
        idx = np.argsort(ss)[::-1]
        ss = ss[idx]
        pp = pp[idx]
        qq = qq[idx]
        rr = rr[idx]
        tt = tt[idx]

        # Keep only unique predictions
        _, idx = np.unique(pp, return_index=True)
        ss = ss[idx]
        pp = pp[idx]
        qq = qq[idx]
        rr = rr[idx]
        tt = tt[idx]

        # Sort them again because numpy sort does something strange
        idx = np.argsort(ss)[::-1]
        ss = ss[idx]
        pp = pp[idx]
        qq = qq[idx]
        rr = rr[idx]
        tt = tt[idx]

        # Extract the highest scores
        scores_merged[encounter] = array_extend(ss, k_max, DEF_VALUES[0])[:k_max]
        predictions_merged[encounter] = array_extend(pp, k_max, DEF_VALUES[1])[:k_max]
        sources_merged[encounter] = array_extend(qq, k_max, DEF_VALUES[2])[:k_max]
        database_iloc_merged[encounter] = array_extend(rr, k_max, DEF_VALUES[3])[:k_max]
        query_iloc_merged[encounter] = array_extend(tt, k_max, DEF_VALUES[4])[:k_max]

    result = EncounterPredictions.__new__(EncounterPredictions)
    result.query = data[0].query
    result.database = data[0].database
    result._set_predictions(scores_merged, predictions_merged, sources_merged, database_iloc_merged, query_iloc_merged)
    return result
