from typing import Iterable, List, Tuple

import numpy as np


def _cell_means(values: np.ndarray, cell_ids: np.ndarray, num_cells: int):
    values = np.asarray(values, dtype=np.float64)
    cells = np.asarray(cell_ids, dtype=np.int64)
    K = int(num_cells)
    counts = np.bincount(cells, minlength=K).astype(np.float64)
    sums = np.bincount(cells, weights=values, minlength=K).astype(np.float64)
    means = np.full((K,), np.nan, dtype=np.float64)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    return means, counts


def worst_cell_accuracy(correct: np.ndarray, cell_ids: np.ndarray, num_cells: int, min_cell: int = 1) -> float:
    means, counts = _cell_means(correct, cell_ids, num_cells)
    valid = counts >= max(int(min_cell), 1)
    if not np.isfinite(means[valid]).any():
        return 0.0
    return float(np.nanmin(means[valid]))


def worst_cell_loss(losses: np.ndarray, cell_ids: np.ndarray, num_cells: int, min_cell: int = 1) -> float:
    means, counts = _cell_means(losses, cell_ids, num_cells)
    valid = counts >= max(int(min_cell), 1)
    if not np.isfinite(means[valid]).any():
        return 0.0
    return float(np.nanmax(means[valid]))


def between_total_ratio(values: np.ndarray, cell_ids: np.ndarray, num_cells: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.var(ddof=0))
    if total <= 0:
        return 0.0
    means, counts = _cell_means(values, cell_ids, num_cells)
    nz = counts > 0
    if not nz.any():
        return 0.0
    weights = counts[nz] / counts[nz].sum()
    mu = float(values.mean())
    between = float(np.sum(weights * (means[nz] - mu) ** 2))
    return float(between / total)


def aggregate_proxy_metrics(
    losses: np.ndarray,
    correct: np.ndarray,
    partitions: Iterable[np.ndarray],
    num_cells: int,
    min_cell: int = 1,
) -> Tuple[float, float, float, float]:
    worst_accs: List[float] = []
    worst_losses: List[float] = []
    between_loss: List[float] = []
    between_correct: List[float] = []
    for cells in partitions:
        worst_accs.append(worst_cell_accuracy(correct, cells, num_cells, min_cell=min_cell))
        worst_losses.append(worst_cell_loss(losses, cells, num_cells, min_cell=min_cell))
        between_loss.append(between_total_ratio(losses, cells, num_cells))
        between_correct.append(between_total_ratio(correct, cells, num_cells))
    return (
        float(np.mean(worst_accs)) if worst_accs else 0.0,
        float(np.mean(worst_losses)) if worst_losses else 0.0,
        float(np.mean(between_loss)) if between_loss else 0.0,
        float(np.mean(between_correct)) if between_correct else 0.0,
    )


def snr_between_total_multi(
    correct: np.ndarray,
    partitions: Iterable[np.ndarray],
    num_cells: int,
    null_trials: int = 25,
    seed: int = 0,
) -> float:
    correct = np.asarray(correct, dtype=np.float64)
    parts = [np.asarray(p, dtype=np.int64) for p in partitions]
    if correct.size == 0 or len(parts) == 0:
        return 0.0

    obs = []
    for p in parts:
        obs.append(between_total_ratio(correct, p, num_cells))
    observed = float(np.mean(obs))

    rng = np.random.default_rng(int(seed))
    null_vals = []
    for _ in range(int(null_trials)):
        shuf = rng.permutation(correct)
        vals = [between_total_ratio(shuf, p, num_cells) for p in parts]
        null_vals.append(float(np.mean(vals)))
    null_mean = float(np.mean(null_vals)) if null_vals else 0.0
    if null_mean <= 0:
        return 0.0
    return float(observed / null_mean)
