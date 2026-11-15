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
from . import parse

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

def simulate_seen(hz, sit, half, start_seen=True, seed=0, scale_loss=1.0, scale_regain=1.0):
    """Frame-by-frame visibility from the hazards. sit: situation() of the CLEAN frames table, where
    the ball is known everywhere. scale_* multiply the hazards' odds (mess levels)."""
    rng = np.random.default_rng(seed)
    n = len(half)
    Xl = sit[LOSS_FEATURES].fillna(hz["loss"]["medians"])
    p_loss = hz["loss"]["model"].predict_proba(Xl)[:, 1]
    p_loss = _scale_odds(p_loss, scale_loss)
    # the regain hazard also depends on the gap's age, known only during the run: precompute it for
    # every frame on a grid of ages (dense for the first 2 s, log-spaced up to 60 s) and look it up
    reg_cols = [c for c in REGAIN_FEATURES if c != "log_gap_s"]
    Xr = sit[reg_cols].fillna(hz["regain"]["medians"]).to_numpy()
    ages = np.unique(np.r_[np.arange(1, 50), np.round(np.geomspace(50, 60 * FPS, 40))]).astype(int)
    grid = np.empty((len(ages), n), np.float32)
    model = hz["regain"]["model"]
    for k, a in enumerate(ages):
        Xa = pd.DataFrame(np.c_[np.full(n, np.log1p(a / FPS)), Xr], columns=REGAIN_FEATURES)
        grid[k] = _scale_odds(model.predict_proba(Xa)[:, 1], scale_regain)
    age_idx = np.searchsorted(ages, np.arange(int(ages[-1]) + 1)).clip(0, len(ages) - 1)

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
            seen[i] = u[i] < p
            age = 0 if seen[i] else age + 1
    return seen


def _scale_odds(p, s):
    if s == 1.0:
        return p
    o = p / (1 - p) * s
    return o / (1 + o)


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
