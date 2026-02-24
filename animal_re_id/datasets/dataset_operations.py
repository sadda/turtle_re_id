import numpy as np

# class Dataset:
#     def get_subsets(self, matching_parts):
#         return {m: self.get_subset(m) for m in matching_parts}

#     def keep_only_unknown(self):
#         mask = self.dataset.metadata["identity"] == "unknown"
#         self.dataset = self.dataset.get_subset(mask)

#     def remove_unknown(self):
#         mask = self.dataset.metadata["identity"] != "unknown"
#         self.dataset = self.dataset.get_subset(mask)


def get_subset(dataset, matching_part):
    labels = dataset.matching_parts[matching_part]['labels']
    mask = dataset.dataset.metadata['label'].isin(labels)
    return dataset.dataset.get_subset(mask)

def get_ignore_pairs(dataset, matching_part, **kwargs):
    dataset_wd = get_subset(dataset, matching_part)
    cols_equal = dataset.matching_parts[matching_part]['cols_equal']
    cols_unequal = dataset.matching_parts[matching_part]['cols_unequal']
    return _get_ignore_pairs(dataset_wd.metadata, cols_equal, cols_unequal, **kwargs)

def _get_ignore_pairs(df, cols_equal, cols_unequal, ignore_unknown=True):
    ignore_pairs = []

    # Ignores same values in cols_equal
    for col in cols_equal:
        s = df[col]
        if ignore_unknown and col == 'identity':
            s = s[s != 'unknown']
        
        groups = s.groupby(s).groups
        for idx in groups.values():
            idx = idx.to_numpy()
            i, j = np.meshgrid(idx, idx, indexing='ij')
            ignore_pairs.extend(zip(i.ravel(), j.ravel()))

    # Ignores different values in cols_unequal
    for col in cols_unequal:
        s = df[col]
        if ignore_unknown and col == 'identity':
            s = s[s != 'unknown']

        values = s.to_numpy()
        idx = s.index.to_numpy()

        mask = values[:, None] != values[None, :]
        i, j = np.where(mask)
        ignore_pairs.extend(zip(idx[i], idx[j]))

    return ignore_pairs

