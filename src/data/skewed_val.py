import numpy as np


def make_skewed_val_indices(group_ids, size, worst_group_id, worst_group_frac, seed=0):
    """Sample a skewed validation subset by group id.

    Returns (indices, info_dict).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(group_ids))
    worst_idx = idx[group_ids == worst_group_id]
    other_idx = idx[group_ids != worst_group_id]

    size = int(size)
    worst_n = int(round(size * float(worst_group_frac)))
    worst_n = min(worst_n, int(worst_idx.size))
    other_n = size - worst_n
    other_n = min(other_n, int(other_idx.size))

    actual_size = int(worst_n + other_n)
    if actual_size <= 0:
        raise ValueError("Skewed val size resolved to 0; check size and group counts.")

    worst_sel = rng.choice(worst_idx, size=worst_n, replace=False) if worst_n > 0 else np.array([], dtype=np.int64)
    other_sel = rng.choice(other_idx, size=other_n, replace=False) if other_n > 0 else np.array([], dtype=np.int64)
    sel = np.concatenate([worst_sel, other_sel]).astype(np.int64)
    rng.shuffle(sel)

    info = {
        "requested_size": int(size),
        "actual_size": int(actual_size),
        "worst_group_id": int(worst_group_id),
        "worst_group_frac_target": float(worst_group_frac),
        "worst_group_count": int(worst_idx.size),
        "other_group_count": int(other_idx.size),
        "worst_selected": int(worst_n),
        "other_selected": int(other_n),
    }
    return sel, info
