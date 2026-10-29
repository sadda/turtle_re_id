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
