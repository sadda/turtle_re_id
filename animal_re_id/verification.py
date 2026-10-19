import networkx as nx
import pandas as pd

ANSWER_SAME = "same"
ANSWER_DIFFERENT = "different"


def answers_to_groups(answers: pd.DataFrame, col1: str = "identity1", col2: str = "identity2") -> list[tuple]:
    mask = answers["answer"] == ANSWER_SAME
    graph = nx.Graph()
    graph.add_edges_from(answers.loc[mask, [col1, col2]].itertuples(index=False, name=None))
    return [tuple(sorted(component)) for component in nx.connected_components(graph)]


def find_conflicts(answers: pd.DataFrame, groups: list[tuple], col1: str = "identity1", col2: str = "identity2") -> pd.DataFrame:
    group_of = {identity: i for i, group in enumerate(groups) for identity in group}
    mask = answers["answer"] == ANSWER_DIFFERENT
    different = answers.loc[mask]
    in_same_group = [u in group_of and group_of.get(u) == group_of.get(v) for u, v in zip(different[col1], different[col2])]
    return different[in_same_group]


def answers_to_assignments(answers: pd.DataFrame, col_query: str = "identity1", col_database: str = "identity2") -> tuple[dict, pd.DataFrame]:
    same = answers.loc[answers["answer"] == ANSWER_SAME, [col_query, col_database]].drop_duplicates()
    n_matches = same.groupby(col_query)[col_database].transform("nunique")
    conflicts = same[n_matches > 1]
    assignments = dict(same[n_matches == 1].itertuples(index=False, name=None))
    return assignments, conflicts
