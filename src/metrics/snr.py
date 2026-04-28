import numpy as np


def snr_between_total(values, cell_ids, null_trials=50, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    total = np.var(values)
    if total == 0:
        return 0.0
    def between(vals, cells):
        uniq = np.unique(cells)
        means = []
        weights = []
        for c in uniq:
            v = vals[cells == c]
            means.append(np.mean(v))
            weights.append(len(v))
        means = np.array(means)
        weights = np.array(weights) / np.sum(weights)
        return np.sum(weights * (means - np.sum(weights * means)) ** 2)

    observed = between(values, cell_ids) / total
    nulls = []
    for _ in range(null_trials):
        shuf = rng.permutation(values)
        nulls.append(between(shuf, cell_ids) / total)
    return float(observed / (np.mean(nulls) + 1e-8))
