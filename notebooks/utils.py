import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Tuple, Callable

def compute_predictions(
        features_query: np.ndarray,
        features_database: np.ndarray,
        ignore: Optional[List[List[int]]] = None,
        matcher: Callable = cosine_similarity,
        k: int = 4,
        return_score: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a closest match in the database for each vector in the query set.

    Args:
        features_query (np.ndarray): Query features of size n_query*n_feature. 
        features_database (np.ndarray): Database features of size n_database*n_feature
        ignore (Optional[List[List[int]]], optional): `ignore[i]` is a list of indices
            in the database ignores for i-th query.
        matcher (Callable, optional): function computing similarity.
        k (int, optional): Returned number of predictions.
        return_score (bool, optional): Whether the similalarity is returned.

    Returns:
        Vector of size (n_query,) and array of size (n_query,k). The latter are indices
            in the database for the closest matches (with ignored `ignore` indices).
            If `return_score`, it also returns an array of size (n_query,k) of scores.
    """

    # Create batch chunks
    n_query = len(features_query)
    # If ignore is not provided, initialize as empty
    if ignore is None:
        ignore = [[] for _ in range(n_query)]
    
    idx_true = np.array(range(n_query))
    idx_pred = np.zeros((n_query, k), dtype=np.int32)
    scores = np.zeros((n_query, k))
    # Compute the cosine similarity between the query and the database
    similarity = matcher(features_query, features_database)
    # Set -infinity for ignored indices
    for i in range(len(ignore)):
        similarity[i, ignore[i]] = -np.inf
    # Find the closest matches (k highest values)
    idx_pred = (-similarity).argsort(axis=-1)[:, :k]
    if return_score:
        scores = np.take_along_axis(similarity, idx_pred, axis=-1)
        return idx_true, idx_pred, scores
    else:
        return idx_true, idx_pred

def check_datasets(dataset_query, dataset_database):
    for dataset in [dataset_query, dataset_database]:
        if not np.array_equal(dataset.df.index, range(len(dataset))):
            raise ValueError('Dataset index must be 0..n')
    individuals_query = dataset_query.df['identity'].unique()
    individuals_database = dataset_database.df['identity'].unique()
    if not all(np.isin(individuals_query, individuals_database)):
        raise ValueError('All individuals in query must be in the database')

def predictions_labels(similarity, dataset_query, dataset_database, label, k=1):
    check_datasets(dataset_query, dataset_database)
    idx_pred = (-similarity).argsort(axis=-1)
    similarity_sorted = np.take_along_axis(similarity, idx_pred, axis=-1)

    y_true = []
    y_pred = []
    labels = []
    scores = []
    for label_id, df_sighting in dataset_query.df.groupby(label):
        jj = idx_pred[df_sighting.index]
        ss = similarity_sorted[df_sighting.index]
        
        identity_pred = dataset_database.df['identity'].to_numpy()[jj].flatten()
        ss = ss.flatten()
        idx_sort = np.argsort(ss)[::-1]

        y_true.append(df_sighting['identity'].iloc[0])
        y_pred.append(identity_pred[idx_sort[:k]])
        labels.append(label_id)
        scores.append(ss[idx_sort[:k]])

    return np.array(y_true), np.array(y_pred), np.array(labels), np.array(scores)

def compute_counts(y_true, y_pred, similarity):
    similarity_max = np.max(similarity, axis=1)
    hits = np.array(y_true == y_pred[:,0], dtype=int)

    thresholds = np.linspace(0, 1, 100)
    similarity_correct = similarity_max[hits == 1]
    count_correct = [np.sum(similarity_correct >= t) for t in thresholds]
    similarity_wrong = similarity_max[hits == 0]
    count_wrong = [np.sum(similarity_wrong >= t) for t in thresholds]

    return thresholds, count_correct, count_wrong

def predictions_images(similarity, dataset_query, dataset_database, k=1):
    return predictions_labels(similarity, dataset_query, dataset_database, 'image_id', k=k)

def predictions_encounters(similarity, dataset_query, dataset_database, k=1):
    return predictions_labels(similarity, dataset_query, dataset_database, 'encounter_id', k=k)

def top_k_accuracy(y_true, y_pred, k=1):
    return np.mean([x in y[:k] for x, y in zip(y_true, y_pred)])
