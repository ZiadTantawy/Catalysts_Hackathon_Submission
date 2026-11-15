"""Model inputs on top of the ETL frames table: longer context, and the home/away mirror.

    df = prep.add_long_context(df)          # past / next 5 s and 15 s means of key columns
    dfm = prep.swap_home_away(df)           # the same moment seen with home and away exchanged

The mirror is how the model is kept from learning "home = Fortuna Düsseldorf": every training frame is
also used mirrored, and at prediction time the normal and mirrored predictions are averaged.
Every column falls in one group (by its base name, before any "__" window suffix):

- h_* / a_* pairs: exchanged;
- team-signed columns (+1 home / -1 away / 0 none, positive favours home): sign flipped;
- directions on the pitch (the ETL orients home to attack +x): the pitch is rotated 180 degrees,
  so x -> PITCH_L - x, y -> PITCH_W - y, velocities change sign;
- everything else (distances to lines, counts, flags, overall speed): unchanged.
Labels: H <-> A, D unchanged. tests in check_swap() rebuild a match with the teams relabelled through
the ETL and compare it with swap_home_away of the original.
"""

import numpy as np
import pandas as pd

from ..etl import config as C

SIGNED = {"d1_diff", "closer_team", "contact_team", "control_team", "last_contact_team",
          "last_control_team", "off_near_team"}   # closer_run is an unsigned run length: unchanged
X_POS = {"ball_x", "ball_last_x", "ball_next_x"}
Y_POS = {"ball_y", "ball_last_y", "ball_next_y"}
Y_PAIRS = {"cy"}       # team columns in pitch y (not the team's own frame): exchanged AND mirrored
NEGATE = {"ball_vx", "ball_vy", "flow_x"}

LONG_BASES = ["closer_team", "d1_diff", "last_control_team", "last_contact_team", "ball_observed",
              "all_mean_speed", "ball_speed", "flow_x", "h_cx_att", "a_cx_att", "h_vx_att", "a_vx_att",
              "speed_ratio_1s_10s"]
LONG_WINDOWS = {"5s": 5 * C.FPS, "15s": 15 * C.FPS}


def add_long_context(df):
    """Trailing (__p) and leading (__n) means over 5 s and 15 s, inside each half."""
    out = {}
    g = df.groupby("half", sort=False)
    for col in LONG_BASES:
        for name, w in LONG_WINDOWS.items():
            past = g[col].transform(lambda x: x.astype(np.float32).rolling(w, min_periods=1).mean())
            nxt = g[col].transform(lambda x: x.astype(np.float32)[::-1].rolling(w, min_periods=1).mean()[::-1])
            out["{}__p{}".format(col, name)] = past.astype(np.float32)
            out["{}__n{}".format(col, name)] = nxt.astype(np.float32)
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def _partner(col):
    if col.startswith("h_"):
        return "a_" + col[2:]
    if col.startswith("a_"):
        return "h_" + col[2:]
    return None


def swap_home_away(df, label_col="label"):
    """A copy of df with home and away exchanged (see module docstring)."""
    out = df.copy()
    for col in df.columns:
        base = col.split("__")[0]
        p = _partner(col)
        if p is not None and p in df.columns:
            out[col] = (C.PITCH_W - df[p].to_numpy()) if base[2:] in Y_PAIRS else df[p].to_numpy()
        elif base in SIGNED:
            out[col] = -df[col]
        elif base in X_POS:
            out[col] = C.PITCH_L - df[col]
        elif base in Y_POS:
            out[col] = C.PITCH_W - df[col]
        elif base in NEGATE:
            out[col] = -df[col]
    if label_col in df.columns:
        lab = df[label_col].to_numpy()
        out[label_col] = np.where(lab == 0, 1, np.where(lab == 1, 0, lab)).astype(lab.dtype)
    return out


def check_swap(mid="J03WOH"):
    """End-to-end test of the mapping: relabel the teams in the clean tracking (home <-> away), run the
    ETL again, and compare with swap_home_away of the original frames. Returns the worst columns."""
    from . import parse, to_atd
    m = parse.load_match(mid, quiet=True)
    atd = to_atd.clean_arrays(m)
    a, _ = to_atd.build_frames(atd)
    atd_sw = dict(atd, slot_team=1 - atd["slot_team"])
    b, _ = to_atd.build_frames(atd_sw)
    sw = swap_home_away(a, label_col=None)
    rows = []
    for col in a.columns:
        if not pd.api.types.is_numeric_dtype(a[col]) or col in ("frame", "half", "phase", "t_half_s"):
            continue
        x, y = sw[col].to_numpy(np.float64), b[col].to_numpy(np.float64)
        both = np.isfinite(x) & np.isfinite(y)
        nan_mismatch = float((np.isfinite(x) != np.isfinite(y)).mean())
        close = float(np.isclose(x[both], y[both], atol=1e-3, rtol=1e-4).mean()) if both.any() else 1.0
        rows.append({"column": col, "share_equal": close, "nan_mismatch": nan_mismatch})
    return pd.DataFrame(rows).sort_values(["share_equal", "nan_mismatch"], ascending=[True, False])
