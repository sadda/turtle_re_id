import numpy as np


def get_subset(dataset, matching_part):
    labels = dataset.matching_parts[matching_part]["labels"]
    mask = dataset.dataset.metadata["label"].isin(labels)
    return dataset.dataset.get_subset(mask)


def get_ignore_pairs(df1, df2, cols_equal, cols_unequal, ignore_unknown=True):
    ignore_pairs = []

    # Ignores same values in cols_equal
    for col in cols_equal:
        if col not in df1.columns or col not in df2.columns:
            continue
        s1 = df1[col]
        s2 = df2[col]
        if ignore_unknown and col == "identity":
            s1 = s1[s1 != "unknown"]
            s2 = s2[s2 != "unknown"]

        groups2 = s2.groupby(s2).groups
        for val, idx1 in s1.groupby(s1).groups.items():
            if val not in groups2:
                continue
            idx2 = groups2[val]

            i, j = np.meshgrid(idx1.to_numpy(), idx2.to_numpy(), indexing="ij")
            ignore_pairs.extend(zip(i.ravel(), j.ravel()))

    # Ignores different values in cols_unequal
    for col in cols_unequal:
        if col not in df1.columns or col not in df2.columns:
            continue
        s1 = df1[col]
        s2 = df2[col]
        if ignore_unknown and col == "identity":
            s1 = s1[s1 != "unknown"]
            s2 = s2[s2 != "unknown"]

        values1 = s1.to_numpy()
        values2 = s2.to_numpy()
        idx1 = s1.index.to_numpy()
        idx2 = s2.index.to_numpy()

        mask = values1[:, None] != values2[None, :]
        i, j = np.where(mask)
        ignore_pairs.extend(zip(idx1[i], idx2[j]))

    return ignore_pairs
