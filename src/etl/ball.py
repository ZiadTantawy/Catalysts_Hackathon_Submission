"""Stage 4 and stage 8: clean the ball track, and turn it into discrete ball events.

Cleaning is conservative on purpose. Three things happen, all flagged:
  * single-frame teleports (implied speed above BALL_MAX_SPEED on the way in AND out, while the
    neighbours agree with each other) are removed;
  * gaps of at most INTERP_MAX_GAP frames inside one half are linearly bridged;
  * nothing longer is invented -- a 41% missing ball stays missing where it is genuinely missing.

Events are candidates, not decisions: an "impulse" is a frame where the ball's velocity changes
abruptly (a kick, a touch, a deflection, a bounce); exits and entries are boundary crossings; lost
and found mark long visibility gaps. Each event carries who was near the ball at that moment.
Deciding which of them are *controlled* touches is the modelling step, not this one.

events.parquet team columns use TEAM_IDX_* (0 home / 1 away / -1 nobody).
"""

import numpy as np
import pandas as pd

from . import config as C
from .util import d_centre, d_goalline, d_touchline, out_dist, runs_true


def _prev_next_index(valid):
    """For every frame, index of the last valid frame <= t and the next valid frame >= t (-1/n)."""
    n = len(valid)
    idx = np.arange(n)
    prev = np.where(valid, idx, -1)
    prev = np.maximum.accumulate(prev)
    nxt = np.where(valid, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    return prev, nxt


def runs_table(cols, half):
    """One row per continuous stretch of observed ball inside a graded half."""
    obs = cols["ball_observed"] & (half > 0)
    st, en = runs_true(obs)
    x, y, sp = cols["ball_x"], cols["ball_y"], cols["ball_speed"]

    rows = []
    for k, (a, b) in enumerate(zip(st, en)):
        h0 = slice(a, min(b, a + 12))
        h1 = slice(max(a, b - 12), b)
        with np.errstate(all="ignore"):
            s0 = float(np.nanmean(sp[h0])) if np.isfinite(sp[h0]).any() else np.nan
            s1 = float(np.nanmean(sp[h1])) if np.isfinite(sp[h1]).any() else np.nan
        rows.append({
            "run_id": k, "start_frame": int(a), "end_frame": int(b), "half": int(half[a]),
            "n_frames": int(b - a),
            "x_start": x[a], "y_start": y[a], "x_end": x[b - 1], "y_end": y[b - 1],
            "speed_start": s0, "speed_end": s1,
            "out_dist_start": float(out_dist(x[a], y[a])),
            "out_dist_end": float(out_dist(x[b - 1], y[b - 1])),
            "d_touchline_start": float(d_touchline(y[a])),
            "d_goalline_start": float(d_goalline(x[a])),
            "d_centre_start": float(d_centre(x[a], y[a])),
            "d_touchline_end": float(d_touchline(y[b - 1])),
            "d_goalline_end": float(d_goalline(x[b - 1])),
            "gap_before": int(a - en[k - 1]) if k > 0 and half[en[k - 1] - 1] == half[a] else -1,
            "gap_after": int(st[k + 1] - b) if k + 1 < len(st) and half[st[k + 1]] == half[a] else -1,
        })
    df = pd.DataFrame(rows)
    run_id = np.full(len(half), -1, dtype=np.int32)
    for k, (a, b) in enumerate(zip(st, en)):
        run_id[a:b] = k
    if len(df):
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
    return df, run_id


def _remove_teleports(x, y, obs, stats):
    """Drop single-frame spikes; flag remaining implausible jumps. Mutates x, y, obs."""
    n = len(obs)
    i = np.flatnonzero(obs)
    spike = np.zeros(n, dtype=bool)
    if len(i) >= 3:
        a, m, z = i[:-2], i[1:-1], i[2:]
        s_in = np.hypot(x[m] - x[a], y[m] - y[a]) / ((m - a) / C.FPS)
        s_out = np.hypot(x[z] - x[m], y[z] - y[m]) / ((z - m) / C.FPS)
        s_skip = np.hypot(x[z] - x[a], y[z] - y[a]) / ((z - a) / C.FPS)
        bad = (s_in > C.BALL_MAX_SPEED) & (s_out > C.BALL_MAX_SPEED) & (s_skip <= C.BALL_MAX_SPEED)
        spike[m[bad]] = True
    x[spike] = np.nan
    y[spike] = np.nan
    obs &= ~spike
    stats["n_spikes_removed"] = int(spike.sum())

    i = np.flatnonzero(obs)
    jump = np.zeros(n, dtype=bool)
    step_speed = None
    if len(i) >= 2:
        step_speed = np.hypot(np.diff(x[i]), np.diff(y[i])) / (np.diff(i) / C.FPS)
        jump[i[1:][step_speed > C.BALL_MAX_SPEED]] = True
    stats["n_jumps_flagged"] = int(jump.sum())
    return obs, jump, i, step_speed


def _bridge_gaps(x, y, obs, half, stats):
    """Linearly fill short gaps inside a half. Returns (xi, yi, fill, prev, nxt, since, until)."""
    n = len(obs)
    prev, nxt = _prev_next_index(obs)
    gap = nxt - prev - 1
    ok_ends = (prev >= 0) & (nxt < n)
    same_half = np.zeros(n, dtype=bool)
    same_half[ok_ends] = half[prev[ok_ends]] == half[nxt[ok_ends]]
    fill = (~obs) & ok_ends & same_half & (gap <= C.INTERP_MAX_GAP)
    # never bridge a gap the ball could not physically have crossed (docs/PLAN.md D5)
    p_, q_ = np.clip(prev, 0, n - 1), np.clip(nxt, 0, n - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        implied = np.hypot(x[q_] - x[p_], y[q_] - y[p_]) / ((nxt - prev) / C.FPS)
    impossible = fill & (implied > C.BALL_MAX_SPEED)
    stats["n_gap_frames_not_bridged"] = int(impossible.sum())
    fill &= ~impossible
    xi, yi = x.copy(), y.copy()
    if fill.any():
        p, q = prev[fill], nxt[fill]
        w = (np.flatnonzero(fill) - p) / (q - p)
        xi[fill] = x[p] + w * (x[q] - x[p])
        yi[fill] = y[p] + w * (y[q] - y[p])
    stats["n_interpolated"] = int(fill.sum())
    idx = np.arange(n)
    since = np.where(prev >= 0, idx - prev, -1).astype(np.int32)
    until = np.where(nxt < n, nxt - idx, -1).astype(np.int32)
    return xi, yi, fill, prev, nxt, since, until


def _ball_velocity(xi, yi):
    k = C.VEL_HALF_WINDOW
    n = len(xi)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vx[k:-k] = (xi[2 * k:] - xi[:-2 * k]) / (2 * k / C.FPS)
    vy[k:-k] = (yi[2 * k:] - yi[:-2 * k]) / (2 * k / C.FPS)
    return vx, vy, np.hypot(vx, vy)


def _unreliable_mask(obs, jump, i, step_speed, stats):
    n = len(obs)
    step_fast = np.zeros(n, dtype=bool)
    if step_speed is not None:
        adj = np.diff(i) == 1
        step_fast[i[1:][(step_speed > C.BALL_MAX_SPEED) & adj]] = True
    unreliable = jump.copy()
    for sh in (1, 2):
        unreliable[sh:] |= jump[:-sh]
        unreliable[:-sh] |= jump[sh:]
    unreliable &= obs
    fs, fe = runs_true(step_fast)
    stats["fast_runs"] = int(len(fs))
    stats["fast_run_len_p50"] = float(np.median(fe - fs)) if len(fs) else 0.0
    return unreliable


def _carry_last_next(x, y, half, prev, nxt):
    n = len(half)

    def carry(idx_arr, ok):
        j = np.clip(idx_arr, 0, n - 1)
        good = ok & (half[j] == half)
        return (np.where(good, x[j], np.nan).astype(np.float32),
                np.where(good, y[j], np.nan).astype(np.float32))

    last_x, last_y = carry(prev, prev >= 0)
    next_x, next_y = carry(nxt, nxt < n)
    return last_x, last_y, next_x, next_y


def _track_diagnostics(x, y, obs, half, stats):
    ii = np.flatnonzero(obs[1:-1] & obs[:-2] & obs[2:]) + 1
    if len(ii):
        d2 = np.hypot(x[ii + 1] - 2 * x[ii] + x[ii - 1], y[ii + 1] - 2 * y[ii] + y[ii - 1])
        st1 = np.hypot(x[ii] - x[ii - 1], y[ii] - y[ii - 1])
        stats["second_diff_cm_p50_p90_p99"] = [float(v * 100) for v in np.percentile(d2, [50, 90, 99])]
        stats["share_exactly_linear"] = float(np.mean((d2 < 1e-4) & (st1 > 1e-4)))
        stats["share_not_moving"] = float(np.mean(st1 < 1e-4))
        stats["step_speed_p50_p90_p99"] = [float(v * C.FPS) for v in np.percentile(st1, [50, 90, 99])]
    vs, ve = runs_true(obs & (half > 0))
    if len(vs):
        ln = (ve - vs) / C.FPS
        stats["visible_runs"] = int(len(vs))
        stats["visible_run_s_p50_p90_max"] = [float(np.median(ln)), float(np.percentile(ln, 90)),
                                              float(ln.max())]


def clean(b_m, half):
    """Return a dict of per-frame ball columns. b_m is [T, >=2] in oriented metres."""
    n = len(b_m)
    x = b_m[:, 0].astype(np.float64)
    y = b_m[:, 1].astype(np.float64)
    obs = np.isfinite(x) & np.isfinite(y)
    stats = {"n_observed_raw": int(obs.sum())}

    obs, jump, i, step_speed = _remove_teleports(x, y, obs, stats)
    xi, yi, fill, prev, nxt, since, until = _bridge_gaps(x, y, obs, half, stats)
    vx, vy, speed = _ball_velocity(xi, yi)
    unreliable = _unreliable_mask(obs, jump, i, step_speed, stats)
    last_x, last_y, next_x, next_y = _carry_last_next(x, y, half, prev, nxt)
    _track_diagnostics(x, y, obs, half, stats)

    cols = {
        "ball_unreliable": unreliable,
        "ball_last_x": last_x, "ball_last_y": last_y,
        "ball_next_x": next_x, "ball_next_y": next_y,
        "ball_x": xi.astype(np.float32),
        "ball_y": yi.astype(np.float32),
        "ball_observed": obs,
        "ball_interpolated": fill,
        "ball_jump_flag": jump,
        "ball_frames_since_seen": since,
        "ball_frames_until_seen": until,
        "ball_vx": vx.astype(np.float32),
        "ball_vy": vy.astype(np.float32),
        "ball_speed": speed.astype(np.float32),
    }
    if b_m.shape[1] >= 3:
        # meaning unknown (docs/PLAN.md D6): kept raw, never turned into a feature
        z = b_m[:, 2].astype(np.float32)
        zk = np.isfinite(z)
        cols["ball_ch3"] = z
        both = zk & obs & np.isfinite(speed)
        stats["third_channel_present"] = float(zk.mean())
        stats["third_channel_without_xy"] = int((zk & ~obs).sum())
        stats["third_channel_corr_with_speed"] = (
            float(np.corrcoef(z[both], speed[both])[0, 1]) if both.sum() > 100 else float("nan"))
        stats["third_channel_p50_p99"] = (
            (float(np.nanpercentile(z, 50)), float(np.nanpercentile(z, 99))) if zk.any() else None)
    return cols, stats


def _rolling_max(a, half_width):
    a0 = np.where(np.isfinite(a), a, 0.0)
    pad = np.pad(a0, half_width, constant_values=0.0)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half_width + 1)
    return win.max(axis=1)


def events(ball_cols, half, near, contact_episodes=()):
    """(event table, noise estimate). `near`: per-frame d_home, d_away, n_home_r, n_away_r.

    contact_episodes: [(start, duration, team TEAM_SIGN_*, min_dist)] from features.contacts().
    """
    n = len(half)
    obs = ball_cols["ball_observed"]
    x = np.where(obs, ball_cols["ball_x"], np.nan).astype(np.float64)
    y = np.where(obs, ball_cols["ball_y"], np.nan).astype(np.float64)
    xi = ball_cols["ball_x"].astype(np.float64)
    yi = ball_cols["ball_y"].astype(np.float64)
    rows = []

    def attrib(t):
        dh, da = near["d_home"][t], near["d_away"][t]
        team = C.TEAM_IDX_NONE
        if np.isfinite(dh) or np.isfinite(da):
            team = (C.TEAM_IDX_HOME if (np.nan_to_num(dh, nan=1e9) <= np.nan_to_num(da, nan=1e9))
                    else C.TEAM_IDX_AWAY)
        return {"nearest_team": team, "d_home": dh, "d_away": da,
                "n_home_at_ball": near["n_home_r"][t], "n_away_at_ball": near["n_away_r"][t]}

    zbest = np.full(n, -np.inf)
    kbest = np.zeros(n, dtype=np.int16)
    store = {}
    noise_info = {}
    for k in C.EVENT_SCALES:
        vbx = np.full(n, np.nan)
        vby = np.full(n, np.nan)
        vax = np.full(n, np.nan)
        vay = np.full(n, np.nan)
        vbx[k:] = (x[k:] - x[:-k]) / (k / C.FPS)
        vby[k:] = (y[k:] - y[:-k]) / (k / C.FPS)
        vax[:-k] = (x[k:] - x[:-k]) / (k / C.FPS)
        vay[:-k] = (y[k:] - y[:-k]) / (k / C.FPS)
        dv = np.hypot(vax - vbx, vay - vby)
        pool = dv[np.isfinite(dv) & (half > 0)]
        if len(pool) > 1000:
            med = float(np.median(pool))
            noise = max(float(1.4826 * np.median(np.abs(pool - med))), 1e-3)
        else:
            med, noise = 0.0, 1.0
        z = np.where(np.isfinite(dv) & (dv > C.EVENT_MIN_DV), (dv - med) / noise, -np.inf)
        better = z > zbest
        zbest[better] = z[better]
        kbest[better] = k
        store[k] = (dv, vbx, vby, vax, vay)
        noise_info["k{}".format(k)] = {
            "dv_median": med, "dv_noise_scale": noise,
            "dv_p90_p99_p999": [float(v) for v in np.percentile(pool, [90, 99, 99.9])]
            if len(pool) else [np.nan] * 3}
    zf = np.where(np.isfinite(zbest), zbest, 0.0)
    peak = (zbest >= C.EVENT_Z_MIN) & (zf >= _rolling_max(zf, C.EVENT_MIN_SEP)) & (half > 0)
    noise_info["z_min"] = C.EVENT_Z_MIN
    noise_info["z_strong"] = C.EVENT_NOISE_K
    for t in np.flatnonzero(peak):
        k = int(kbest[t])
        dv, vbx, vby, vax, vay = store[k]
        sb, sa = np.hypot(vbx[t], vby[t]), np.hypot(vax[t], vay[t])
        ang = np.nan
        if sb > 1.0 and sa > 1.0:
            c = (vbx[t] * vax[t] + vby[t] * vay[t]) / (sb * sa)
            ang = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        rows.append({"frame": t, "kind": "impulse", "dv": dv[t], "dv_z": zbest[t], "scale": k,
                     "strong": bool(zbest[t] >= C.EVENT_NOISE_K),
                     "at_jump": bool(ball_cols["ball_unreliable"][t]), "speed_before": sb,
                     "speed_after": sa, "angle_change": ang, **attrib(t)})

    for start, dur, team, dmin in contact_episodes:
        team_idx = C.TEAM_IDX_HOME if team > 0 else C.TEAM_IDX_AWAY
        row = {"frame": start, "kind": "contact", "duration": dur, "min_dist": dmin,
               "team": team_idx, "strong": dur >= C.CONTROL_MIN, **attrib(start)}
        row["nearest_team"] = row["team"]
        rows.append(row)

    m = C.OOB_MARGIN
    fin = np.isfinite(xi) & np.isfinite(yi)
    out_x = fin & ((xi < -m) | (xi > C.PITCH_L + m))
    out_y = fin & ((yi < -m) | (yi > C.PITCH_W + m))
    inside = fin & ~out_x & ~out_y
    trans = np.flatnonzero(fin[1:] & fin[:-1] & (inside[1:] != inside[:-1]) & (half[1:] > 0)) + 1
    for t in trans:
        side = "goal_line" if out_x[t] or out_x[t - 1] else "touchline"
        kind = "exit" if not inside[t] else "entry"
        rows.append({"frame": t, "kind": kind, "side": side, **attrib(t - 1 if kind == "exit" else t)})

    o = obs.astype(np.int8)
    edges = np.flatnonzero(np.diff(o)) + 1
    starts = edges[o[edges] == 0]
    ends = edges[o[edges] == 1]
    for s in starts:
        j = np.searchsorted(ends, s)
        e = int(ends[j]) if j < len(ends) else n
        g = e - s
        if g >= C.BALL_LOST_MIN_GAP and half[s - 1] > 0:
            rows.append({"frame": s - 1, "kind": "ball_lost", "gap_len": g, **attrib(s - 1)})
            if e < n and half[e] > 0:
                rows.append({"frame": e, "kind": "ball_found", "gap_len": g, **attrib(e)})

    cols = ["frame", "kind", "half", "x", "y", "dv", "dv_z", "scale", "strong", "at_jump", "duration",
            "team", "min_dist", "speed_before", "speed_after",
            "angle_change", "side", "gap_len", "nearest_team", "d_home", "d_away",
            "n_home_at_ball", "n_away_at_ball"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols), noise_info
    df["half"] = half[df["frame"].to_numpy()]
    df["x"] = xi[df["frame"].to_numpy()].astype(np.float32)
    df["y"] = yi[df["frame"].to_numpy()].astype(np.float32)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    df = df[cols].sort_values(["frame", "kind"], kind="mergesort").reset_index(drop=True)
    df["kind"] = df["kind"].astype("category")
    df["side"] = df["side"].astype("string")
    df["strong"] = df["strong"].astype("boolean")
    df["at_jump"] = df["at_jump"].astype("boolean")
    for c in ("dv", "dv_z", "min_dist", "speed_before", "speed_after", "angle_change", "d_home", "d_away"):
        df[c] = df[c].astype(np.float32)
    for c in ("gap_len", "n_home_at_ball", "n_away_at_ball", "scale", "duration", "team"):
        df[c] = df[c].astype("Int32")
    df["nearest_team"] = df["nearest_team"].astype(np.int8)
    df["frame"] = df["frame"].astype(np.int32)
    df["half"] = df["half"].astype(np.int8)
    return df, noise_info
