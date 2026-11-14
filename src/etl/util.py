"""Small pure helpers shared across the ETL (runs, pitch geometry)."""

import numpy as np

from . import config as C


def runs(v):
    """Maximal runs of equal values. Returns (starts, ends_exclusive, values_at_starts)."""
    v = np.asarray(v)
    n = len(v)
    if n == 0:
        return np.array([], int), np.array([], int), v[:0]
    change = np.flatnonzero(v[1:] != v[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    return starts, ends, v[starts]


def runs_true(mask):
    """Maximal True runs of a boolean mask. Returns (starts, ends_exclusive)."""
    st, en, val = runs(np.asarray(mask, dtype=bool))
    keep = val.astype(bool)
    return st[keep], en[keep]


def run_length_signed(s):
    """Length of the current run of identical non-zero values; 0 where s == 0."""
    n = len(s)
    out = np.zeros(n, dtype=np.int32)
    if n == 0:
        return out
    change = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(change)
    out = (np.arange(n) - starts[np.cumsum(change) - 1] + 1).astype(np.int32)
    out[s == 0] = 0
    return out


def out_dist(x, y):
    """Euclidean distance outside the pitch; 0 inside. Accepts scalars or arrays."""
    ox = np.fmax(np.fmax(-x, x - C.PITCH_L), 0.0)
    oy = np.fmax(np.fmax(-y, y - C.PITCH_W), 0.0)
    return np.hypot(ox, oy)


def d_touchline(y):
    return np.fmin(y, C.PITCH_W - y)


def d_goalline(x):
    return np.fmin(x, C.PITCH_L - x)


def d_centre(x, y):
    return np.hypot(x - C.PITCH_L / 2, y - C.PITCH_W / 2)
