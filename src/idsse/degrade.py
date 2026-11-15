"""Make clean IDSSE tracking look like BAR-ATH's automatic tracking (the degrader, plan W2).

Part 1, the ball (data-driven). BAR-ATH shows when its tracker loses the ball; no label is needed,
only whether the ball was seen. Two hazards are learned on BAR-ATH's graded frames:

    loss:   P(ball unseen at t+1 | seen at t, situation at t)
    regain: P(ball seen at t+1 | unseen at t, situation at t, time since it was lost)

The loss hazard may use the ball's own situation (speed, distance to the lines, players around it),
because the ball is visible at that moment in both datasets. The regain hazard only uses what is known
while the ball is hidden in BAR-ATH: how long it has been gone, how fast the players move, and where it
was last seen. On IDSSE the two hazards drive a two-state simulation frame by frame, so gap lengths and
where gaps happen come from BAR-ATH, not from a guess.

Raw "seen" on BAR-ATH = ball_observed and not ball_interpolated (the ETL bridges gaps <= 10 frames;
the same bridging runs again on degraded IDSSE, so the simulation works on raw visibility).
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from ..etl import config as C
from . import camera, parse

FPS = C.FPS

LOSS_FEATURES = ["ball_speed", "ball_d_touchline", "ball_d_goalline", "ball_out_dist", "n_near_3m",
                 "d1_min", "players_speed_3s", "speed_ratio_1s_10s"]
REGAIN_FEATURES = ["log_gap_s", "players_speed_3s", "players_speed_1s", "speed_ratio_1s_10s",
                   "last_d_touchline", "last_d_goalline"]
BAR_COLUMNS = ["graded", "half", "ball_observed", "ball_interpolated", "ball_x", "ball_y", "ball_speed",
               "ball_d_touchline", "ball_d_goalline", "ball_out_dist", "h_n_within_3m", "a_n_within_3m",
               "d1_min", "all_mean_speed__b3s", "all_mean_speed__b1s", "speed_ratio_1s_10s",
               "ball_last_x", "ball_last_y"]


# ---- data --------------------------------------------------------------------------------------

def barath_cache_dir():
    """$BARATH_CACHE, else this repo's cache/, else the etl-v1.3 worktree's cache/ next to it."""
    cands = [os.environ.get("BARATH_CACHE"), C.CACHE_DIR,
             os.path.join(os.path.dirname(parse.REPO_ROOT), "etl-v1.3", "cache")]
    for d in cands:
        if d and os.path.exists(os.path.join(d, "frames.parquet")):
            return d
    raise FileNotFoundError("BAR-ATH frames.parquet not found: build the ETL or set BARATH_CACHE")


def load_barath(columns=BAR_COLUMNS):
    return pd.read_parquet(os.path.join(barath_cache_dir(), "frames.parquet"), columns=columns)


def situation(df, seen):
    """Model inputs for every frame, from an ETL frames table and the raw visibility mask."""
    out = pd.DataFrame(index=df.index)
    out["ball_speed"] = df["ball_speed"]
    out["ball_d_touchline"] = df["ball_d_touchline"]
    out["ball_d_goalline"] = df["ball_d_goalline"]
    out["ball_out_dist"] = df["ball_out_dist"]
    out["n_near_3m"] = df["h_n_within_3m"] + df["a_n_within_3m"]
    out["d1_min"] = df["d1_min"]
    out["players_speed_3s"] = df["all_mean_speed__b3s"]
    out["players_speed_1s"] = df["all_mean_speed__b1s"]
    out["speed_ratio_1s_10s"] = df["speed_ratio_1s_10s"]
    # where the ball was last seen (from the raw mask, so it is the same definition on both datasets)
    half = df["half"].to_numpy()
    lx = pd.Series(np.where(seen, df["ball_x"], np.nan)).groupby(half).ffill().to_numpy()
    ly = pd.Series(np.where(seen, df["ball_y"], np.nan)).groupby(half).ffill().to_numpy()
    out["last_d_touchline"] = np.minimum(ly, C.PITCH_W - ly)
    out["last_d_goalline"] = np.minimum(lx, C.PITCH_L - lx)
    return out


def gap_age(seen, half):
    """Frames since the ball was last seen (0 while seen), restarting each half."""
    age = np.zeros(len(seen), np.int32)
    for i in range(1, len(seen)):
        age[i] = 0 if seen[i] else (age[i - 1] + 1 if half[i] == half[i - 1] else 1)
    return age


# ---- hazards -----------------------------------------------------------------------------------

def _model():
    # smooth, regularised non-linearity: ~150 loss / regain events per match is little data
    return make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=2),
                         LogisticRegression(C=0.3, max_iter=2000))


def _xy(sit, seen, half, graded, kind):
    nxt = np.r_[seen[1:], seen[-1]]
    same = np.r_[half[1:] == half[:-1], False]
    base = graded & same
    if kind == "loss":
        rows = base & seen
        X = sit.loc[rows, LOSS_FEATURES]
        y = ~nxt[rows]
    else:
        rows = base & ~seen
        X = sit.loc[rows, [c for c in REGAIN_FEATURES if c != "log_gap_s"]].copy()
        X.insert(0, "log_gap_s", np.log1p(gap_age(seen, half)[rows] / FPS))
        y = nxt[rows]
    return X.fillna(X.median()), y.astype(int)


def fit_hazards(df, train_mask=None):
    """Fit loss and regain hazards on a BAR-ATH frames table. train_mask limits the frames used."""
    seen = (df["ball_observed"].astype(bool) & ~df["ball_interpolated"].astype(bool)).to_numpy()
    half = df["half"].to_numpy()
    graded = df["graded"].to_numpy().astype(bool)
    if train_mask is not None:
        graded = graded & train_mask
    sit = situation(df, seen)
    out = {}
    for kind in ("loss", "regain"):
        X, y = _xy(sit, seen, half, graded, kind)
        out[kind] = {"model": _model().fit(X, y), "medians": X.median(), "n_events": int(y.sum()),
                     "n_frames": int(len(y)), "base_rate": float(y.mean())}
    return out


def heldout_loglik(df, kind, fit_mask, test_mask):
    """Mean log-likelihood per event of a hazard on test frames, vs a constant hazard (the base rate)."""
    hz = fit_hazards(df, fit_mask)[kind]
    seen = (df["ball_observed"].astype(bool) & ~df["ball_interpolated"].astype(bool)).to_numpy()
    X, y = _xy(situation(df, seen), seen, df["half"].to_numpy(),
               df["graded"].to_numpy().astype(bool) & test_mask, kind)
    p = np.clip(hz["model"].predict_proba(X.fillna(hz["medians"]))[:, 1], 1e-6, 1 - 1e-6)
    q = hz["base_rate"]
    ll = lambda pp: np.sum(y * np.log(pp) + (1 - y) * np.log(1 - pp))
    return {"events": int(y.sum()), "loglik_gain_per_event": (ll(p) - ll(np.full(len(y), q))) / max(y.sum(), 1)}


# ---- simulation --------------------------------------------------------------------------------

def hazard_tables(hz, sit):
    """Per-frame hazards for a situation table (the slow part, done once per match):
    p_loss [T], and p_regain on a grid of gap ages [A, T] plus the age -> grid row lookup."""
    n = len(sit)
    p_loss = hz["loss"]["model"].predict_proba(sit[LOSS_FEATURES].fillna(hz["loss"]["medians"]))[:, 1]
    # the regain hazard also depends on the gap's age, known only during the run: precompute it for
    # every frame on a grid of ages (dense for the first 2 s, log-spaced up to 60 s) and look it up
    reg_cols = [c for c in REGAIN_FEATURES if c != "log_gap_s"]
    Xr = sit[reg_cols].fillna(hz["regain"]["medians"]).to_numpy()
    ages = np.unique(np.r_[np.arange(1, 50), np.round(np.geomspace(50, 60 * FPS, 40))]).astype(int)
    grid = np.empty((len(ages), n), np.float32)
    for k, a in enumerate(ages):
        Xa = pd.DataFrame(np.c_[np.full(n, np.log1p(a / FPS)), Xr], columns=REGAIN_FEATURES)
        grid[k] = hz["regain"]["model"].predict_proba(Xa)[:, 1]
    age_idx = np.searchsorted(ages, np.arange(int(ages[-1]) + 1)).clip(0, len(ages) - 1)
    return {"p_loss": p_loss, "grid": grid, "age_idx": age_idx}


def simulate_seen(hz, sit, half, start_seen=True, seed=0, scale_loss=1.0, scale_regain=1.0,
                  dead=None, tables=None):
    """Frame-by-frame visibility from the hazards. sit: situation() of the frames table with the true
    ball (known everywhere). scale_* multiply the hazards' odds: a number, or a pair (live, dead)
    applied per frame by the `dead` mask (IDSSE's BallStatus), which lets the simulation match how
    BAR-ATH's ball loss splits between live play and stoppages."""
    rng = np.random.default_rng(seed)
    n = len(half)
    tb = tables or hazard_tables(hz, sit)
    dead = np.zeros(n, bool) if dead is None else dead
    s_loss = _per_frame(scale_loss, dead)
    s_reg = _per_frame(scale_regain, dead)
    p_loss = _scale_odds(tb["p_loss"], s_loss)
    grid, age_idx = tb["grid"], tb["age_idx"]

    u = rng.random(n)
    seen = np.ones(n, bool)
    age = 0
    for i in range(n):
        if i == 0 or half[i] != half[i - 1]:
            seen[i], age = start_seen, 0 if start_seen else 1
            continue
        if seen[i - 1]:
            seen[i] = u[i] >= p_loss[i - 1]
            age = 0 if seen[i] else 1
        else:
            p = grid[age_idx[min(age, len(age_idx) - 1)], i - 1]
            if s_reg[i - 1] != 1.0:
                o = p / (1 - p) * s_reg[i - 1]
                p = o / (1 + o)
            seen[i] = u[i] < p
            age = 0 if seen[i] else age + 1
    return seen


def _per_frame(scale, dead):
    live_s, dead_s = scale if isinstance(scale, (tuple, list)) else (scale, scale)
    return np.where(dead, dead_s, live_s).astype(np.float64)


def _scale_odds(p, s):
    o = p / (1 - p) * s
    return o / (1 + o)


# ---- part 2: players (rules, tuned to BAR-ATH's aggregates) --------------------------------------
#
# What the ATD does (docs/PLAN.md, cache/report.txt): a player is measured (speed present) only when
# the camera sees them; otherwise the tracker holds the last position with no speed, and after a while
# drops the track. Players hidden behind others on screen are also unmeasured for a moment. A track
# that comes back gets a new slot (46 slots for 22 players over a match). Targets on graded frames:
# 10.14 / 10.11 tracked per team, 9.27 / 9.48 measured per team, 5.0% of tracked players off screen.

N_HOME_SLOTS, N_AWAY_SLOTS = 24, 22          # BAR-ATH's slot layout


def _nearest_other(X, Y, chunk=5000):
    """Distance from each present position to the nearest other present position, per frame."""
    out = np.full(X.shape, np.inf, np.float32)
    for s in range(0, len(X), chunk):
        x, y = X[s:s + chunk], Y[s:s + chunk]
        d = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        d[:, np.arange(x.shape[1]), np.arange(x.shape[1])] = np.inf
        out[s:s + chunk] = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=2)
    return out


def _last_index(mask, half):
    """Index of the last True at or before each frame within the same half, -1 if none."""
    frame = np.arange(len(mask)).reshape(-1, *[1] * (mask.ndim - 1))
    idx = np.where(mask, frame, -1)
    starts = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1, len(half)]
    for a, b in zip(starts[:-1], starts[1:]):
        idx[a:b] = np.maximum.accumulate(idx[a:b], axis=0)
    return idx


def _hold_limits(unmeas, half, rng, short_s, p_long, long_s):
    """Per position: how many frames into its unmeasured run the tracker still holds it. Each run gets
    its own limit: usually short (exponential, mean short_s), sometimes long (uniform up to long_s)."""
    lim = np.zeros(unmeas.shape, np.int32)
    starts = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1, len(half)]
    for h0, h1 in zip(starts[:-1], starts[1:]):                # runs never cross half-time
        for j in range(unmeas.shape[1]):
            d = np.diff(np.r_[False, unmeas[h0:h1, j], False].astype(np.int8))
            for a, b in zip(np.flatnonzero(d == 1) + h0, np.flatnonzero(d == -1) + h0):
                s = rng.uniform(20, long_s) if rng.random() < p_long else rng.exponential(short_s)
                lim[a:b, j] = int(s * FPS)
    return lim


def degrade_players(atd, short_s=2.0, p_long=0.0, long_s=120.0, occ_r=1.0, height_factor=None,
                    keeper_ghosts=True, fragment=True, seed=0):
    """Camera-limited measurement, held positions, dropped tracks and new slots. Returns a new atd."""
    rng = np.random.default_rng(seed)
    p = atd["players"]
    half = atd["half"]
    cx, cy, w = atd["cam_track"]
    hf = atd["height_factor"] if height_factor is None else height_factor
    X, Y, S = p[..., 0], p[..., 1], p[..., 2]
    present = np.isfinite(X)
    on = present & camera.in_view(X, Y, cx[:, None], cy[:, None], w[:, None], hf)
    meas = on & (_nearest_other(X, Y) >= occ_r)

    last = _last_index(meas, half)                             # [T, S]
    rows = np.arange(len(X))[:, None]
    age = rows - last
    limit = _hold_limits(present & ~meas, half, rng, short_s, p_long, long_s)
    if keeper_ghosts:   # the ATD keeps unmeasured keeper-slot positions for minutes (P6 "ghosts")
        limit[:, atd["slot_is_gk"]] = np.iinfo(np.int32).max
    held = present & ~meas & (last >= 0) & (age <= limit)
    safe = np.where(last >= 0, last, 0)
    cols = np.arange(X.shape[1])[None, :]
    Xo = np.where(meas, X, np.where(held, X[safe, cols], np.nan))
    Yo = np.where(meas, Y, np.where(held, Y[safe, cols], np.nan))
    So = np.where(meas, S, np.nan)

    slot_team = np.r_[np.zeros(N_HOME_SLOTS, np.int64), np.ones(N_AWAY_SLOTS, np.int64)]
    out = np.full((len(X), len(slot_team), 3), np.nan, np.float32)
    slot_is_gk = np.zeros(len(slot_team), bool)
    for t, pool in ((0, np.arange(N_HOME_SLOTS)), (1, N_HOME_SLOTS + np.arange(N_AWAY_SLOTS))):
        segs = []                                              # (start, end, source column)
        for j in np.flatnonzero(atd["slot_team"] == t):
            tr = np.isfinite(Xo[:, j])
            if not tr.any():
                continue
            d = np.diff(np.r_[False, tr, False].astype(np.int8))
            st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            if not fragment:                                   # one slot per person
                st, en = st[:1], en[-1:]
            segs += [(a, b, j) for a, b in zip(st, en)]
        free_at = {s: -1 for s in pool}
        for a, b, j in sorted(segs):
            s = min((s for s in pool if free_at[s] <= a), key=lambda s: free_at[s], default=None)
            if s is None:                                      # more tracks than slots: drop it
                continue
            out[a:b, s, 0], out[a:b, s, 1], out[a:b, s, 2] = Xo[a:b, j], Yo[a:b, j], So[a:b, j]
            free_at[s] = b
            if atd["slot_is_gk"][j] and not slot_is_gk[pool].any():
                slot_is_gk[s] = True
    return {**atd, "players": out, "slot_team": slot_team, "slot_is_gk": slot_is_gk, "height_factor": hf}


def player_stats(atd):
    """Tracked / measured players per team and share of tracked players off screen (all frames)."""
    p, t = atd["players"], atd["slot_team"]
    tracked = np.isfinite(p[..., 0])
    meas = tracked & np.isfinite(p[..., 2])
    cam = camera.features(p, atd["ball"], *atd["cam_track"], atd["height_factor"])
    return {"tracked_home": round(float(tracked[:, t == 0].sum(1).mean()), 2),
            "tracked_away": round(float(tracked[:, t == 1].sum(1).mean()), 2),
            "measured_home": round(float(meas[:, t == 0].sum(1).mean()), 2),
            "measured_away": round(float(meas[:, t == 1].sum(1).mean()), 2),
            "measured_%": round(100 * float(meas.sum() / max(tracked.sum(), 1)), 1),
            "off_screen_%": round(100 * float(np.nanmean(cam["frac_tracked_off_screen"])), 1)}


# ---- both together ---------------------------------------------------------------------------------

def degrade(atd, hz, dead, seed=0, scale_loss=(0.5, 3.0), scale_regain=(2.5, 0.2), **player_kw):
    """Players first, then the ball: the ball hazards read player-based inputs (players around the
    ball, their speed), which on BAR-ATH come from degraded tracking, so they must be degraded on IDSSE
    too before the hazards are applied.

    dead: IDSSE's BallStatus == 0 per frame. The hazards' odds are scaled separately in live play and
    in stoppages (scale_* = (live, dead)), fitted so the ball is missing on ~80% of dead and ~11% of
    live frames, as on BAR-ATH by its own dead-ball detector (no labels; 2026-11-15: 79.7% / 11.2%).

    Not done: hiding the ball whenever it leaves the virtual camera's view. The real camera keeps the
    ball in view (99.9% of frames where it is known); the virtual one follows the players more than
    the ball, so that rule doubled the number of gaps (tested 2026-11-15)."""
    from . import to_atd                                       # to_atd imports camera, not degrade
    d = degrade_players(atd, seed=seed, **player_kw)
    df, _ = to_atd.build_frames(d)
    seen = simulate_seen(hz, situation(df, np.ones(len(df), bool)), d["half"], seed=seed,
                         scale_loss=scale_loss, scale_regain=scale_regain, dead=dead)
    b = d["ball"]
    ball_out = b.copy()
    ball_out[~seen] = np.nan
    return {**d, "ball": ball_out, "seen": seen}


# ---- statistics (the calibration targets) --------------------------------------------------------

def gap_stats(seen, half, graded=None):
    """Visible share and missing-gap lengths, in the same terms as cache/report.txt."""
    graded = np.ones(len(seen), bool) if graded is None else graded
    lens = []
    for h in np.unique(half[graded]):
        s = seen[graded & (half == h)]
        miss = np.r_[False, ~s, False]
        d = np.diff(miss.astype(np.int8))
        lens += list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    lens = np.array(lens) / FPS
    miss_frames = lens.sum() * FPS
    return {"visible_%": round(100 * seen[graded].mean(), 1), "gaps": len(lens),
            "gaps_per_90min": round(len(lens) / (graded.sum() / FPS / 60) * 90, 1),
            "gap_s_p50": round(float(np.median(lens)), 1) if len(lens) else np.nan,
            "gap_s_p90": round(float(np.percentile(lens, 90)), 1) if len(lens) else np.nan,
            "missing_in_gaps_>5s_%": round(100 * lens[lens > 5].sum() * FPS / max(miss_frames, 1), 1)}
