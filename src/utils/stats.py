from __future__ import annotations

import math
from typing import Iterable

import numpy as np


_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def t_critical_95(df: int) -> float:
    df = int(df)
    if df <= 0:
        return 0.0
    if df in _T975:
        return float(_T975[df])
    return 1.96


def ci95_mean(x: Iterable[float] | np.ndarray) -> float:
    arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n <= 1:
        return 0.0
    return float(t_critical_95(n - 1) * arr.std(ddof=1) / math.sqrt(float(n)))


def cvar_top_fraction(vals: Iterable[float] | np.ndarray, q: float) -> float:
    arr = np.asarray(list(vals) if not isinstance(vals, np.ndarray) else vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    q = float(q)
    if q <= 0.0:
        return float(arr.mean())
    if q >= 1.0:
        return float(arr.max())
    k = max(1, int(math.ceil(q * float(arr.size))))
    idx = np.argpartition(arr, arr.size - k)[arr.size - k :]
    return float(arr[idx].mean())
