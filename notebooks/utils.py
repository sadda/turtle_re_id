import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Tuple, Callable
from turtlewatch_egypt import TWE_NewData

def compute_counts(y_true, y_pred, similarity):
    similarity_max = np.max(similarity, axis=1)
    hits = np.array(y_true == y_pred[:,0], dtype=int)

    thresholds = np.linspace(0, 1, 100)
    similarity_correct = similarity_max[hits == 1]
    count_correct = [np.sum(similarity_correct >= t) for t in thresholds]
    similarity_wrong = similarity_max[hits == 0]
    count_wrong = [np.sum(similarity_wrong >= t) for t in thresholds]

    return thresholds, count_correct, count_wrong

def top_k_accuracy(y_true, y_pred, k=1):
    return np.mean([x in y[:k] for x, y in zip(y_true, y_pred)])
