import numpy as np


def flattening_index(values, cell_ids):
    """Between/total variance ratio."""
    values = np.asarray(values)
    total = np.var(values)
    if total == 0:
        return 0.0
    uniq = np.unique(cell_ids)
    means = []
    weights = []
    for c in uniq:
        v = values[cell_ids == c]
        means.append(np.mean(v))
        weights.append(len(v))
    means = np.array(means)
    weights = np.array(weights) / np.sum(weights)
    between = np.sum(weights * (means - np.sum(weights * means)) ** 2)
    return float(between / total)
