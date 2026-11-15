"""Stage 7b: whole-pitch situations that mean the ball is dead, as per-frame flags.

Each is evidence for the labeller, never a label on its own, and each is evaluated on graded frames
only. The graded window is not exactly whistle to whistle: it starts before the 2nd-half kickoff and
runs past the 1st-half whistle, so walk-offs can be flagged on graded frames. That is correct, because
those frames are dead ball. Decisions in docs/PLAN.md.

  known_dead       (D8)  second half graded before its kickoff (published in FAQ.md)
  h_/a_surplus     (P7)  more than 11 measured players of one team, sustained (celebrations,
                         confrontations, substitutions, staff on the pitch)
  track_blackout   (P10) measured players collapse below half their recent level, or stay below
                         BLACKOUT_ABS for seconds (close-ups, tracking loss)
  dense_cluster    (D9)  a dense, slow cluster of both teams within 5 m (set-piece crowds)
  crowd            (D9)  most of the tracked players in one slow, mixed group within 10 m
                         (confrontations; a share, because the tracker loses people inside a crowd)
  cooling_break    (D9)  a long stoppage in which most players leave tracking and those visible
                         group near a touchline
  ball_still_long  (D7)  ball stationary for seconds: a restart being set up
  ball_isolated    (D7)  ball slow and far from every measured player: spare ball or false detection
"""

import warnings

import numpy as np
import pandas as pd

from . import config as C
from .util import runs_true


def _keep_long(mask, min_len):
    out = np.zeros_like(mask)
    st, en = runs_true(mask)
    for a, b in zip(st, en):
        if b - a >= min_len:
            out[a:b] = True
    return out


def _close_gaps(mask, max_gap):
    """Fill False gaps of at most max_gap frames between two True runs."""
    out = mask.copy()
    st, en = runs_true(mask)
    for i in range(1, len(st)):
        if st[i] - en[i - 1] <= max_gap:
            out[en[i - 1]:st[i]] = True
    return out


def _dense_cluster(X, Y, team, speed1s, radius):
    """Per frame: size of the densest `radius` neighbourhood, minority-team share, median speed."""
    n, s = X.shape
    size = np.zeros(n, dtype=np.int16)
    minority = np.zeros(n, dtype=np.float32)
    cl_speed = np.full(n, np.nan, dtype=np.float32)
    for a0 in range(0, n, C.PAIR_CHUNK):
        a1 = min(n, a0 + C.PAIR_CHUNK)
        x, y, tm = X[a0:a1], Y[a0:a1], team[a0:a1]
        ok = np.isfinite(x) & (tm >= 0)
        d = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        nb = (d < radius) & ok[:, :, None] & ok[:, None, :]
        cnt = nb.sum(2)
        c = np.argmax(cnt, axis=1)
        idx = np.arange(a1 - a0)
        mem = nb[idx, c]
        nh = (mem & (tm == C.TEAM_IDX_HOME)).sum(1)
        na = (mem & (tm == C.TEAM_IDX_AWAY)).sum(1)
        size[a0:a1] = cnt[idx, c]
        minority[a0:a1] = np.minimum(nh, na) / np.maximum(nh + na, 1)
        sp = np.where(mem, speed1s[a0:a1], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cl_speed[a0:a1] = np.nanmedian(sp, axis=1)
    return size, minority, cl_speed


def compute(pl, feat, ball_cols, tl):
    """Return (columns, stats). pl from players.prepare(), feat from features, tl from orient."""
    n = len(tl["half"])
    graded = tl["graded"]
    X, Y, team, meas = pl["X"], pl["Y"], pl["team"], pl["meas"]
    kept = team >= 0
    cols, stats = {}, {}

    known_dead = np.zeros(n, dtype=bool)
    a2 = C.HALF_BLOCKS[1][1]
    known_dead[a2:C.KICKOFF_2ND] = True
    cols["known_dead"] = known_dead

    for t, pre in ((C.TEAM_IDX_HOME, "h_"), (C.TEAM_IDX_AWAY, "a_")):
        many = graded & ((kept & meas & (team == t)).sum(1) > 11)
        cols[pre + "surplus"] = _keep_long(many, C.SURPLUS_MIN)

    n_meas = (kept & meas).sum(1).astype(float)
    base = pd.Series(n_meas).rolling(C.SLOW_WINDOW, min_periods=50).median().to_numpy()
    drop = graded & np.isfinite(base) & (n_meas < C.BLACKOUT_RATIO * base)
    low = graded & (n_meas < C.BLACKOUT_ABS)
    cols["track_blackout"] = _keep_long(drop, C.BLACKOUT_MIN) | _keep_long(low, C.BLACKOUT_ABS_MIN)

    # dense, slow, mixed cluster (set-piece crowds, confrontations)
    step = np.full(X.shape, np.nan, dtype=np.float32)
    step[1:] = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0)) * C.FPS
    speed1s = pd.DataFrame(step).rolling(C.WINDOWS["1s"], min_periods=5, center=True) \
        .median().to_numpy(dtype=np.float32)
    size, minority, cl_speed = _dense_cluster(X, Y, team, speed1s, C.CLUSTER_R)
    dense = graded & (size >= C.CLUSTER_N) & (minority >= C.CLUSTER_MINORITY) \
        & (cl_speed < C.CLUSTER_SPEED)
    cols["dense_cluster_n"] = size.astype(np.int8)
    cols["dense_cluster"] = _keep_long(dense, C.CLUSTER_MIN)

    # crowd: the same test at CROWD_R, as a share of the tracked players
    size_r, minority_r, speed_r = _dense_cluster(X, Y, team, speed1s, C.CROWD_R)
    n_kept = kept.sum(1)
    share = np.where(n_kept > 0, size_r / np.maximum(n_kept, 1), 0.0)
    crowd = graded & (n_kept >= C.CROWD_MIN_TRACKED) & (share >= C.CROWD_SHARE) \
        & (minority_r >= C.CLUSTER_MINORITY) & (speed_r < C.CLUSTER_SPEED)
    cols["crowd_n"] = size_r.astype(np.int8)
    cols["crowd_share"] = share.astype(np.float32)
    cols["crowd"] = _keep_long(crowd, C.CROWD_MIN)

    # cooling break: long, slow stoppage, most players gone from tracking, the rest by a touchline
    slow = pd.Series(feat["all_mean_speed"]).rolling(3 * C.FPS, min_periods=25, center=True) \
        .mean().to_numpy()
    yk = np.where(kept, Y, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cy = np.nanmean(yk, axis=1)
    by_line = np.minimum(cy, C.PITCH_W - cy) <= C.COOL_TOUCHLINE
    cand = graded & (slow < C.COOL_SPEED) & (kept.sum(1) <= C.COOL_MAX_TRACKED) & by_line
    cand = _close_gaps(cand, C.COOL_MERGE) & graded
    cols["cooling_break"] = _keep_long(cand, C.COOL_MIN)

    # ball states
    stat = feat["ball_stationary"] == 1
    cols["ball_still_long"] = _keep_long(stat & ball_cols["ball_observed"], C.BALL_STILL_LONG)
    bsp = pd.Series(ball_cols["ball_speed"]).rolling(12, min_periods=3, center=True).median().to_numpy()
    iso = graded & ball_cols["ball_observed"] & (np.nan_to_num(feat["d1_min"], nan=np.inf) > C.BALL_ISOLATED_R) \
        & (bsp < C.BALL_ISOLATED_SPEED)
    cols["ball_isolated"] = _keep_long(iso, C.BALL_ISOLATED_MIN)

    for k, v in cols.items():
        if v.dtype == bool:
            st, en = runs_true(v & graded)
            stats[k] = {"frames": int((v & graded).sum()), "stretches": int(len(st)),
                        "longest": sorted(((int(b - a), int(a)) for a, b in zip(st, en)),
                                          reverse=True)[:3]}
    return cols, stats
