import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
