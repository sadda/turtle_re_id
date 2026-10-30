import networkx as nx
import numpy as np
import pandas as pd

ANSWER_SAME = "same"
ANSWER_DIFFERENT = "different"
PREDICTION_CONFLICT = "conflict"
PREDICTION_NEW = "new"
PREDICTION_UNKNOWN = None


def generate_answers(verification: pd.DataFrame, true_identities: dict) -> pd.DataFrame:
    answers = verification.copy()
    answers["time"] = 0.0
    answers["answer"] = ""
    for encounter, answers_encounter in answers.groupby("encounter1"):
        identity = true_identities[encounter]
        index = answers_encounter.index
        correct = np.where(answers_encounter["identity2"] == identity)[0]
        if len(correct) > 0:
            i = correct[0]
            answers.loc[index[:i], "answer"] = ANSWER_DIFFERENT
            answers.loc[index[i], "answer"] = ANSWER_SAME
        else:
            answers.loc[index, "answer"] = ANSWER_DIFFERENT
    return answers


def generate_answers_discovery(verification: pd.DataFrame, true_identities: dict) -> pd.DataFrame:
    answers = verification.copy()
    answers["time"] = 0.0
    answers["answer"] = ""
    pairs_different = []
    for i, row in answers.iterrows():
        encounter1 = row["encounter1"]
        encounter2 = row["encounter2"]

        # Groups of encounters already answered as same
        groups = answers_to_groups(answers)
        group1 = groups.loc[groups["encounters"].map(lambda e: encounter1 in e), "encounters"].iloc[0]
        group2 = groups.loc[groups["encounters"].map(lambda e: encounter2 in e), "encounters"].iloc[0]

        # Skip if the answer is already known
        if encounter2 in group1:
            continue
        is_known_different = False
        for a, b in pairs_different:
            if (a in group1 and b in group2) or (a in group2 and b in group1):
                is_known_different = True
        if is_known_different:
            continue

        if true_identities[encounter1] == true_identities[encounter2]:
            answers.loc[i, "answer"] = ANSWER_SAME
        else:
            answers.loc[i, "answer"] = ANSWER_DIFFERENT
            pairs_different.append((encounter1, encounter2))
    return answers


def answers_to_predictions(answers: pd.DataFrame, col_encounter: str = "encounter1", col_identity: str = "identity2") -> pd.DataFrame:
    rows = []
    for encounter, answers_encounter in answers.groupby(col_encounter, sort=False):
        is_same = answers_encounter["answer"] == ANSWER_SAME
        is_different = answers_encounter["answer"] == ANSWER_DIFFERENT
        identities_same = answers_encounter.loc[is_same, col_identity].unique()
        if len(identities_same) == 1:
            prediction = identities_same[0]
        elif len(identities_same) > 1:
            prediction = PREDICTION_CONFLICT
        elif is_different.all():
            prediction = PREDICTION_NEW
        else:
            prediction = PREDICTION_UNKNOWN
        rows.append({"encounter": encounter, "prediction": prediction})
    return pd.DataFrame(rows)


def answers_to_groups(answers: pd.DataFrame, encounters: list | None = None, col_encounter1: str = "encounter1", col_encounter2: str = "encounter2") -> pd.DataFrame:
    graph = nx.Graph()
    if encounters is not None:
        graph.add_nodes_from(encounters)
    graph.add_nodes_from(answers[col_encounter1])
    graph.add_nodes_from(answers[col_encounter2])
    is_same = answers["answer"] == ANSWER_SAME
    graph.add_edges_from(answers.loc[is_same, [col_encounter1, col_encounter2]].to_numpy().tolist())

    rows = []
    for encounters_group in nx.connected_components(graph):
        rows.append({"encounters": sorted(encounters_group)})
    groups = pd.DataFrame(rows)
    groups = groups.sort_values("encounters", key=lambda s: s.str[0], ignore_index=True)
    return groups
