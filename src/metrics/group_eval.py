from typing import Dict, Tuple

import numpy as np


def group_id_from_y_a(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    a_max = int(a.max()) if a.size else 0
    return (y * (a_max + 1) + a).astype(np.int64)


def group_accuracy_from_logits(
    logits: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
) -> Tuple[float, float, Dict[int, float], Dict[int, int]]:
    logits = np.asarray(logits, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    g = np.asarray(group_ids, dtype=np.int64)
    preds = (logits >= 0).astype(np.int64)
    correct = (preds == y).astype(np.float32)

    overall_acc = float(correct.mean()) if correct.size else 0.0

    uniq = np.unique(g)
    acc_map: Dict[int, float] = {}
    cnt_map: Dict[int, int] = {}
    worst = 1.0
    for gid in uniq:
        mask = g == gid
        cnt = int(mask.sum())
        if cnt <= 0:
            continue
        acc = float(correct[mask].mean())
        acc_map[int(gid)] = acc
        cnt_map[int(gid)] = cnt
        worst = min(worst, acc)

    worst_acc = float(worst if acc_map else 0.0)
    return overall_acc, worst_acc, acc_map, cnt_map

