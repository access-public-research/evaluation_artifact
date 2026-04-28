import numpy as np


def _rank_bins(x: np.ndarray, k: int) -> np.ndarray:
    n = int(x.size)
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    if k <= 1:
        return np.zeros((n,), dtype=np.int64)
    order = np.argsort(x, kind="mergesort")
    bins = np.empty((n,), dtype=np.int64)
    bin_ids = (np.arange(n, dtype=np.int64) * int(k)) // int(n)
    bins[order] = bin_ids
    return bins


def random_proj_bins(embeddings: np.ndarray, num_cells: int, seed: int = 0, return_vector: bool = False):
    rng = np.random.default_rng(int(seed))
    d = int(embeddings.shape[1])
    r = rng.standard_normal(d).astype(np.float64)
    r = r / (np.linalg.norm(r) + 1e-12)
    proj = (embeddings.astype(np.float64) @ r).astype(np.float64)
    bins = _rank_bins(proj, int(num_cells))
    if return_vector:
        return bins, r
    return bins
