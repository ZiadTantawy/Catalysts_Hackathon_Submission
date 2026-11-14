"""Stages 5-7 and 10: per-frame team, contest and dead-ball features, plus context windows.

Conventions after orient.py: metres, home attacks +x. Team features are expressed in each team's
own attacking frame (x_att = x for home, L - x for away), so "h_highest_att" and "a_highest_att"
mean the same thing for both sides and a model can treat the teams symmetrically.

Which positions are used where (docs/PLAN.md D1-D3, P9), from players.prepare():
  * team membership and the goalkeeper role are per frame, never taken from the slot number;
  * everything about the ball (distances, nearest player, contacts, control) uses MEASURED players
    only -- a guessed position cannot be on the ball;
  * team shape and tempo use every kept position, measured or guessed, where a few metres of error
    do not matter;
  * when the nearest measured player belongs to an opposite-team one-object pair, the contest gives
    no team (contested_pair).

Column prefixes:  h_ home team · a_ away team · ball_ ball · (none) both / contest / dead-ball.
Signed contest columns use TEAM_SIGN_* (+1 home / -1 away / 0 none); positive favours HOME.
"""

import warnings

import numpy as np
import pandas as pd

from . import config as C
from .util import d_centre, d_goalline, d_touchline, out_dist, run_length_signed, runs


def _nan_reduce(fn, a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def speed_and_frozen(p):
    """Derived speed (the tracker's where present, else from positions) and frozen frames.

    The derived speed is for tempo features only. Whether a position was measured is decided from
    the tracker's own speed channel in players.prepare(), before anything is filled here.
    """
    X, Y = p[..., 0], p[..., 1]
    spd_pos = np.full_like(X, np.nan)
    spd_pos[1:] = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0)) * C.FPS
    if p.shape[2] >= 3:
        spd = p[..., 2].copy()
        miss = ~np.isfinite(spd) & np.isfinite(X)
        spd[miss] = spd_pos[miss]
        n_filled = int(miss.sum())
    else:
        spd, n_filled = spd_pos, int(np.isfinite(spd_pos).sum())

    n = X.shape[0]
    same = np.ones(n, dtype=bool)
    same[0] = False
    fin = np.isfinite(X)
    same[1:] = (fin[1:] == fin[:-1]).all(1) & fin[1:].any(1)
    same[1:] &= np.where(fin[1:], X[1:] == X[:-1], True).all(1)
    same[1:] &= np.where(fin[1:], Y[1:] == Y[:-1], True).all(1)
    return spd, same, {"speed_derived_from_positions": n_filled, "frozen_frames": int(same.sum())}


def _slot_velocity_x(X, Y):
    """Per-track velocity along x; jumps faster than SLOT_VEL_MAX are discarded."""
    k = C.SLOT_VEL_K
    VX = np.full(X.shape, np.nan, dtype=np.float32)
    dxk = (X[2 * k:] - X[:-2 * k]) / (2 * k / C.FPS)
    dyk = (Y[2 * k:] - Y[:-2 * k]) / (2 * k / C.FPS)
    with np.errstate(invalid="ignore"):
        VX[k:-k] = np.where(np.hypot(dxk, dyk) <= C.SLOT_VEL_MAX, dxk, np.nan)
    return VX


def _first_two(dd):
    """Smallest and second-smallest per row of an inf-filled array, NaN where absent."""
    if dd.shape[1] >= 2:
        part = np.partition(dd, 1, axis=1)
        d1, d2 = part[:, 0], part[:, 1]
    else:
        d1, d2 = dd[:, 0], np.full(len(dd), np.inf)
    return np.where(np.isfinite(d1), d1, np.nan), np.where(np.isfinite(d2), d2, np.nan)


def _team_shape_tempo(cols, near, pl, SPD, VX, D, n):
    """Per-team shape, tempo, keeper and distance-to-ball columns (h_ then a_)."""
    X, Y, team, meas, keeper, off = pl["X"], pl["Y"], pl["team"], pl["meas"], pl["keeper"], pl["off"]
    rows = np.arange(n)
    for t, pre in ((C.TEAM_IDX_HOME, "h_"), (C.TEAM_IDX_AWAY, "a_")):
        m = team == t                                       # kept positions of this team
        of = m & ~keeper & ~off                             # outfield, on the pitch: team shape
        att = (lambda a: a) if t == C.TEAM_IDX_HOME else (lambda a: C.PITCH_L - a)
        xa = np.where(of, att(X), np.nan)
        y = np.where(of, Y, np.nan)
        vxa = np.where(m, VX if t == C.TEAM_IDX_HOME else -VX, np.nan)
        cols[pre + "vx_att"] = _nan_reduce(np.nanmean, vxa)
        cols[pre + "n_off_pitch"] = (m & off).sum(1).astype(np.int8)
        s = np.where(m, SPD, np.nan)
        cnt = np.isfinite(xa).sum(1).astype(np.int8)
        cx = _nan_reduce(np.nanmean, xa)
        cols[pre + "n_tracked"] = m.sum(1).astype(np.int8)
        cols[pre + "n_measured"] = (m & meas).sum(1).astype(np.int8)
        cols[pre + "n_outfield"] = cnt
        cols[pre + "cx_att"] = cx
        cols[pre + "cy"] = _nan_reduce(np.nanmean, y)
        cols[pre + "spread"] = np.sqrt(_nan_reduce(np.nanvar, xa) + _nan_reduce(np.nanvar, y))
        cols[pre + "deepest_att"] = _nan_reduce(np.nanmin, xa)
        cols[pre + "highest_att"] = _nan_reduce(np.nanmax, xa)
        cols[pre + "length"] = cols[pre + "highest_att"] - cols[pre + "deepest_att"]
        cols[pre + "width"] = _nan_reduce(np.nanmax, y) - _nan_reduce(np.nanmin, y)
        cols[pre + "mean_speed"] = _nan_reduce(np.nanmean, s)
        cols[pre + "max_speed"] = _nan_reduce(np.nanmax, s)
        cols[pre + "own_half_frac"] = np.where(
            cnt > 0, (xa < C.PITCH_L / 2 + 1.0).sum(1) / np.maximum(cnt, 1), np.nan)
        vel = np.full(n, np.nan)
        vel[5:-5] = (cx[10:] - cx[:-10]) / (10 / C.FPS)
        cols[pre + "cx_att_vel"] = vel

        gk = m & keeper                                     # per-frame role (D3), at most one
        cols[pre + "gk_x_att"] = _nan_reduce(np.nanmin, np.where(gk, att(X), np.nan))
        cols[pre + "gk_d_ball"] = _nan_reduce(np.nanmin, np.where(gk, D, np.nan))

        # everything about the ball: measured players only (D2)
        mm = m & meas
        dd = np.where(mm & np.isfinite(D), D, np.inf)
        d1, d2 = _first_two(dd)
        cols[pre + "d1_ball"] = d1
        cols[pre + "d2_ball"] = d2
        nearest = np.argmin(dd, axis=1)
        cols[pre + "nearest_speed"] = np.where(np.isfinite(d1), SPD[rows, nearest], np.nan)
        dm = np.where(mm, D, np.nan)
        for r in C.NEAR_RADII:
            cols["{}n_within_{:g}m".format(pre, r)] = (dm <= r).sum(1).astype(np.int8)
        cols[pre + "n_within_crowd"] = (dm <= C.CROWD_RADIUS).sum(1).astype(np.int8)
        near["d_home" if t == C.TEAM_IDX_HOME else "d_away"] = d1
        near["n_home_r" if t == C.TEAM_IDX_HOME else "n_away_r"] = (dm <= C.EVENT_NEAR_R).sum(1)


def _contest(cols, contested, n):
    """Who is closer to the ball, and for how long. No team where the nearest is a one-object pair."""
    diff = cols["a_d1_ball"] - cols["h_d1_ball"]
    cols["d1_diff"] = diff
    closer = np.zeros(n, dtype=np.int8)
    closer[np.isfinite(diff) & (diff > 0)] = C.TEAM_SIGN_HOME
    closer[np.isfinite(diff) & (diff < 0)] = C.TEAM_SIGN_AWAY
    only_h = np.isfinite(cols["h_d1_ball"]) & ~np.isfinite(cols["a_d1_ball"])
    only_a = np.isfinite(cols["a_d1_ball"]) & ~np.isfinite(cols["h_d1_ball"])
    closer[only_h] = C.TEAM_SIGN_HOME
    closer[only_a] = C.TEAM_SIGN_AWAY
    closer[contested] = C.TEAM_SIGN_NONE
    cols["closer_team"] = closer
    cols["contested_pair"] = contested
    cols["closer_run"] = run_length_signed(closer)
    cols["d1_min"] = np.fmin(cols["h_d1_ball"], cols["a_d1_ball"])


def _deadball_cues(cols, SPD, kept, ball_cols, n):
    """Tempo ratios, kickoff shape, and ball out-of-bounds / stationary geometry."""
    allspd = _nan_reduce(np.nanmean, np.where(kept, SPD, np.nan))
    cols["n_tracked"] = kept.sum(1).astype(np.int8)
    cols["all_mean_speed"] = allspd
    s_fast = pd.Series(allspd).rolling(C.WINDOWS["1s"], min_periods=5).mean().to_numpy()
    s_slow = pd.Series(allspd).rolling(C.SLOW_WINDOW, min_periods=25).mean().to_numpy()
    cols["speed_ratio_1s_10s"] = s_fast / np.where(s_slow > 0.05, s_slow, np.nan)
    cols["kickoff_shape"] = np.where(
        (cols["h_n_tracked"] >= 8) & (cols["a_n_tracked"] >= 8),
        np.fmin(cols["h_own_half_frac"], cols["a_own_half_frac"]), np.nan)

    bxv, byv = ball_cols["ball_x"], ball_cols["ball_y"]
    ball_known = np.isfinite(bxv)
    m = C.OOB_MARGIN
    oob = np.full(n, -1, dtype=np.int8)
    oob[ball_known] = 0
    oob[ball_known & ((byv < -m) | (byv > C.PITCH_W + m))] = 1
    oob[ball_known & ((bxv < -m) | (bxv > C.PITCH_L + m))] = 2
    cols["ball_oob"] = oob
    cols["ball_d_touchline"] = d_touchline(byv)
    cols["ball_d_goalline"] = d_goalline(bxv)
    cols["ball_d_centre"] = d_centre(bxv, byv)
    cols["ball_out_dist"] = out_dist(bxv, byv)
    sp = pd.Series(ball_cols["ball_speed"]).rolling(
        C.STATIONARY_WINDOW, min_periods=6).mean().to_numpy()
    stat = np.full(n, -1, dtype=np.int8)
    stat[np.isfinite(sp)] = (sp[np.isfinite(sp)] < C.STATIONARY_SPEED).astype(np.int8)
    cols["ball_stationary"] = stat


def team_contest_deadball(p, pl, ball_cols):
    """p: [T, S, C] oriented metres; pl: players.prepare() output.

    Returns (columns, near dict for events, stats).
    """
    n = p.shape[0]
    SPD, frozen, hstats = speed_and_frozen(p)
    X, Y, team, meas = pl["X"], pl["Y"], pl["team"], pl["meas"]
    kept = team >= 0
    Xk = np.where(kept, X, np.nan)
    Yk = np.where(kept, Y, np.nan)

    bx = ball_cols["ball_x"][:, None]
    by = ball_cols["ball_y"][:, None]
    D = np.hypot(Xk - bx, Yk - by)
    VX = _slot_velocity_x(Xk, Yk)

    cols, near = {"frozen_frame": frozen}, {}
    cols["flow_x"] = _nan_reduce(np.nanmean, VX)
    off = pl["off"] & kept & meas
    dd_off = np.where(off & np.isfinite(D), D, np.inf)
    j_off = np.argmin(dd_off, axis=1)
    d_off = dd_off[np.arange(n), j_off]
    cols["off_d1_ball"] = np.where(np.isfinite(d_off), d_off, np.nan)
    t_off = team[np.arange(n), j_off]
    cols["off_near_team"] = np.where(
        np.isfinite(d_off),
        np.where(t_off == C.TEAM_IDX_HOME, C.TEAM_SIGN_HOME, C.TEAM_SIGN_AWAY),
        C.TEAM_SIGN_NONE).astype(np.int8)
    cols["n_off_pitch"] = (pl["off"] & kept).sum(1).astype(np.int8)

    _team_shape_tempo(cols, near, pl, SPD, VX, D, n)

    # nearest measured player overall: part of a one-object opposite-team pair -> contested (P9)
    dm = np.where(kept & meas & np.isfinite(D), D, np.inf)
    j = np.argmin(dm, axis=1)
    contested = np.isfinite(dm[np.arange(n), j]) & pl["xdup"][np.arange(n), j]
    _contest(cols, contested, n)
    _deadball_cues(cols, SPD, kept, ball_cols, n)

    hstats["contested_frames"] = int(contested.sum())
    for k, v in list(cols.items()):
        if v.dtype == np.float64:
            cols[k] = v.astype(np.float32)
    return cols, near, hstats, D


def _ffill_within_half(v, half):
    """Carry the last non-zero value forward, restarting at 0 at the start of each half."""
    n = len(v)
    idx = np.where(v != 0, np.arange(n), -1)
    seg_start = np.r_[True, half[1:] != half[:-1]]
    marker = np.where(seg_start, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    reset = np.maximum.accumulate(marker)
    out = np.where(last >= reset, v[np.clip(last, 0, n - 1)], 0).astype(np.int8)
    out[last < 0] = 0
    since = np.where((last >= reset) & (last >= 0), np.arange(n) - last, -1).astype(np.int32)
    out[half == 0] = 0
    since[half == 0] = -1
    return out, since


def _ball_reaction(ball_cols):
    """|v(t+k) - v(t-k)|: how much the ball's velocity changes around each frame (m/s)."""
    k = C.CONTACT_REACT_K
    vx, vy = ball_cols["ball_vx"].astype(np.float64), ball_cols["ball_vy"].astype(np.float64)
    out = np.full(len(vx), np.nan)
    out[k:-k] = np.hypot(vx[2 * k:] - vx[:-2 * k], vy[2 * k:] - vy[:-2 * k])
    return out


def contacts(cols, half, ball_cols):
    """Contact and control from ball-to-nearest-measured-player distance, with a ball reaction.

    Signed +1 home / -1 away / 0 none (TEAM_SIGN_*). A run of frames with one team's measured player
    within CONTACT_R is a contact only if the ball stays close for CONTACT_HOLD frames or visibly
    reacts (D4); a ball flying past someone is not a touch. Episodes for the event table use the same
    signed team; events() maps them to TEAM_IDX_*.
    """
    n = len(half)
    d = cols["d1_min"]
    closer = cols["closer_team"].astype(np.int8)
    team = np.where(np.isfinite(d) & (d <= C.CONTACT_R) & (half > 0), closer, 0).astype(np.int8)

    st, en, val = runs(team)
    for i in range(1, len(st) - 1):
        if val[i] == 0 and en[i] - st[i] <= C.CONTACT_GAP_TOL and val[i - 1] == val[i + 1] != 0 \
                and half[st[i] - 1] == half[en[i]]:
            team[st[i]:en[i]] = val[i - 1]

    react = _ball_reaction(ball_cols)
    st, en, val = runs(team)
    contact = np.zeros(n, dtype=np.int8)
    control = np.zeros(n, dtype=np.int8)
    episodes = []
    n_flyby = 0
    for a, b, v in zip(st, en, val):
        if v == 0 or b - a < C.CONTACT_MIN:
            continue
        lo, hi = max(0, a - 2), min(n, b + 2)
        reacted = np.isfinite(react[lo:hi]).any() and np.nanmax(react[lo:hi]) >= C.CONTACT_REACT_DV
        if b - a < C.CONTACT_HOLD and not reacted:
            n_flyby += 1
            continue
        contact[a:b] = v
        if b - a >= C.CONTROL_MIN:
            control[a:b] = v
        episodes.append((int(a), int(b - a), int(v), float(np.nanmin(d[a:b]))))

    out = {"contact_team": contact, "control_team": control}
    out["last_contact_team"], out["frames_since_contact"] = _ffill_within_half(contact, half)
    out["last_control_team"], out["frames_since_control"] = _ffill_within_half(control, half)
    return out, episodes, {"contact_runs_rejected_flyby": n_flyby}


def context_windows(df):
    """Rolling means over 1 s and 3 s, centred (__c) and trailing (__b), of a few key columns."""
    out = {}
    for col in C.CONTEXT_COLUMNS:
        if col not in df:
            continue
        s = df[col].astype(np.float32)
        for name, w in C.WINDOWS.items():
            out["{}__c{}".format(col, name)] = s.rolling(w, min_periods=1, center=True).mean()
            out["{}__b{}".format(col, name)] = s.rolling(w, min_periods=1).mean()
    return pd.DataFrame(out, index=df.index).astype(np.float32)
