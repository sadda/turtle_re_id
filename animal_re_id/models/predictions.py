import numpy as np
from wildlife_tools.inference import TopkClassifier

DEF_VALUES = [-np.inf, "", ""]


class EncounterPredictions:
    def __init__(self, similarity, query, database, encounter_col: str = "encounter_id", identity_col: str = "identity") -> None:
        groups = query.metadata.groupby(encounter_col).indices
        self.keys = list(groups.keys())
        # TODO: this recomputes it every time
        self.similarity_grouped = np.vstack([similarity[idx].max(axis=0) for idx in groups.values()])
        self.identities_database = database.metadata[identity_col].to_numpy()

    def classify(self, k_max: int, matching_part: str) -> None:
        classifier = TopkClassifier(self.identities_database, k=k_max, return_all=True)
        predictions, scores, _ = classifier(self.similarity_grouped)
        scores_dict = dict(zip(self.keys, scores))
        predictions_dict = dict(zip(self.keys, predictions))
        self._set_predictions(scores_dict, predictions_dict, matching_part, k_max)

    def _set_predictions(self, scores: dict, predictions: dict, matching_parts: str | dict, k_max=None) -> None:
        self.scores = scores
        self.predictions = predictions

        self._save_matching_part(matching_parts)
        if k_max is not None:
            self._extend_data(k_max)

    def _extend_data(self, k_max):
        data = [self.scores, self.predictions, self.matching_parts]
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


def array_extend(x, n, fill_value):
    n_missing = n - len(x)
    if n_missing > 0:
        return np.concatenate((x, np.full(n_missing, fill_value)))
    return x


def merge_data(data: dict[str, EncounterPredictions], k_max) -> EncounterPredictions:
    # Get all keys (encounters)
    matching_parts = data.keys()
    scores = [data[m].scores for m in matching_parts]
    predictions = [data[m].predictions for m in matching_parts]
    encounters = set().union(*scores)

    # Find the index for the matching part with the largest score
    scores_merged = {}
    predictions_merged = {}
    parts_merged = {}
    for encounter in encounters:
        ss = np.vstack([s.get(encounter, np.full(k_max, DEF_VALUES[0])) for s in scores]).flatten()
        pp = np.vstack([s.get(encounter, np.full(k_max, DEF_VALUES[1])) for s in predictions]).flatten()
        qq = np.vstack([np.full(k_max, matching_part) for matching_part in matching_parts]).flatten()

        # Sort the values starting with highest scores
        idx = np.argsort(ss)[::-1]
        ss = ss[idx]
        pp = pp[idx]
        qq = qq[idx]

        # Keep only unique predictions
        _, idx = np.unique(pp, return_index=True)
        ss = ss[idx]
        pp = pp[idx]
        qq = qq[idx]

        # Sort them again because numpy sort does something strange
        idx = np.argsort(ss)[::-1]
        ss = ss[idx]
        pp = pp[idx]
        qq = qq[idx]

        # Extract the highest scores
        scores_merged[encounter] = array_extend(ss, k_max, DEF_VALUES[0])[:k_max]
        predictions_merged[encounter] = array_extend(pp, k_max, DEF_VALUES[1])[:k_max]
        parts_merged[encounter] = array_extend(qq, k_max, DEF_VALUES[2])[:k_max]

    result = EncounterPredictions.__new__(EncounterPredictions)
    result._set_predictions(scores_merged, predictions_merged, parts_merged)
    return result
