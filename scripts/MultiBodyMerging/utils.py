import matplotlib.pyplot as plt
import numpy as np


def get_info(dataset, splitters):
    identities = {}
    for encounter_id, df_encounter in dataset.metadata.groupby("encounter_id"):
        identity = df_encounter["identity"].unique().tolist()
        if len(identity) != 1:
            raise ValueError("Multiple identities for encounter")
        identities[encounter_id] = identity[0]

    n_encounters = {}
    for splitter in splitters:
        split = splitter.get_split(dataset.metadata)
        n_encounters[split.name] = dataset.metadata.loc[split.mask_query, "encounter_id"].nunique()

    return identities, n_encounters


def plot_best_scores(best_scores, matching_parts, y_max=None, use_title=True):
    best_scores_arr = np.asarray(best_scores, dtype=object)

    plt.figure(figsize=(12, 2))
    for i, m in enumerate(matching_parts):
        for correct, color in zip([True, False], ["green", "red"]):
            mask1 = best_scores_arr[:, 0] == correct
            mask2 = best_scores_arr[:, 2] == m
            mask = mask1 & mask2

            plt.subplot(1, len(matching_parts), i + 1)
            plt.hist(best_scores_arr[mask, 1], alpha=0.6, color=color)
            if y_max is not None:
                plt.ylim([0, y_max])
            if use_title:
                plt.title(m)
            plt.grid(alpha=0.6)


def plot_accuracy_B(accuracy, matching_parts, figsize=None, ylim=None, ylabel=None):
    # TODO: handle the colors properly
    colors = ["green", "red", "blue", "orange", "purple"]
    ylim = [0, 1] if ylim is None else ylim

    plt.figure(figsize=figsize)
    for i, matching_part in enumerate(matching_parts + ["merged"]):
        plt.subplot(1, len(matching_parts) + 1, i + 1)
        for name, row in accuracy.loc[matching_part].iterrows():
            linestyle = "-" if "-" in name else "--"
            plt.plot(row.index, row.values, linestyle, color=colors[i], label=name)
        # plt.legend()
        plt.title(matching_part)
        plt.grid(alpha=0.7)
        plt.xlabel("B")
        plt.ylim(ylim)
        if i == 0 and ylabel is not None:
            plt.ylabel(ylabel)
