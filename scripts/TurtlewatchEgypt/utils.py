import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEF_VALUES = [-np.inf, "", ""]


class Data:
    def __init__(self, scores: dict, predictions: dict, matching_parts: str | dict, k_max=None):
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


def merge_data(data: dict[str, Data], k_max) -> Data:
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

    return Data(scores_merged, predictions_merged, parts_merged)


def get_info(dataset, split_years):
    summary = []
    for encounter_id, df_encounter in dataset.metadata.groupby("encounter_id"):
        identity = df_encounter["identity"].unique().tolist()
        year = df_encounter["year"].unique().tolist()
        if len(identity) != 1:
            raise ValueError("Multiple identities for encounter")
        if len(year) != 1:
            raise ValueError("Multiple years for encounter")
        summary.append({"encounter": encounter_id, "identity": identity[0], "year": year[0]})

    identities = {s["encounter"]: s["identity"] for s in summary}
    summary_df = pd.DataFrame(summary)
    n_encounters_all = {}
    n_encounters_known = {}
    for split_year in split_years:
        mask_now = summary_df["year"] == split_year
        mask_before = summary_df["year"] < split_year

        summary_now = summary_df[mask_now]
        summary_before = summary_df[mask_before]

        identities_known = summary_before["identity"].unique()
        mask_known = summary_now["identity"].isin(identities_known)
        summary_now_known = summary_now[mask_known]

        n_encounters_all[split_year] = summary_now["encounter"].nunique()
        n_encounters_known[split_year] = summary_now_known["encounter"].nunique()

    return identities, n_encounters_all, n_encounters_known


def plot_best_scores(best_scores, matching_parts, y_max=None):
    best_scores_arr = np.asarray(best_scores, dtype=object)

    plt.figure(figsize=(10, 2))
    for i, m in enumerate(matching_parts):
        for correct, color in zip([True, False], ["green", "red"]):
            mask1 = best_scores_arr[:, 0] == correct
            mask2 = best_scores_arr[:, 2] == m
            mask = mask1 & mask2

            plt.subplot(1, len(matching_parts), i + 1)
            plt.hist(best_scores_arr[mask, 1], alpha=0.6, color=color)
            if y_max is not None:
                plt.ylim([0, y_max])
            plt.title(m)
            plt.grid(alpha=0.6)
