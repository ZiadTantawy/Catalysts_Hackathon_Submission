"""Training tables: degraded IDSSE matches through the etl-v1.3 features, with variant B labels.

    df = dataset.match_table("J03WOH", level="realistic", seed=1)   # cached in cache/idsse/tables/
    X = df[dataset.feature_columns(df)]

Levels scale the fitted ball-loss dials (degrade.degrade defaults = "realistic", fitted to BAR-ATH),
so a model trained on all three does not depend on the degrader being exactly right. Player
degradation is the same at every level (it matched BAR-ATH's counts closely).

Columns a model must not use (feature_columns drops them):
- bookkeeping: frame, timeline, run ids;
- Smart Tagging (`tag_*`, set-piece and goal-tag timing): IDSSE has none;
- camera (`cam_*`, on-screen counts): simulated on IDSSE, so not the same signal as BAR-ATH's;
- `known_dead` (a BAR-ATH-only published fact), `ball_ch3` (unknown ATD channel), labels.
"""

import os

import numpy as np
import pandas as pd

from . import bench, degrade, parse, to_atd

TABLE_DIR = os.path.join(parse.CACHE_DIR, "tables")
LEVELS = {                     # multipliers on degrade.degrade's (live, dead) odds scales
    "light": {"loss": 0.6, "regain": 1.6},
    "realistic": {"loss": 1.0, "regain": 1.0},
    "heavy": {"loss": 1.6, "regain": 0.6},
}
LABEL_COLUMNS = ["label", "dead", "match", "level", "seed"]
EXCLUDE_PREFIXES = ("tag_", "cam_", "setpiece", "frames_since_setpiece", "frames_until_setpiece",
                    "next_setpiece", "frames_since_goal_tag")
EXCLUDE = {"frame", "half", "phase", "graded", "t_half_s", "rotated", "known_dead", "ball_ch3",
           "ball_run_id", "n_on_screen", "frac_tracked_off_screen", "ball_on_screen"}

_HZ = None


def hazards():
    global _HZ
    if _HZ is None:
        _HZ = degrade.fit_hazards(degrade.load_barath())
    return _HZ


def _scaled(pair, k):
    return tuple(s * k for s in pair)


def match_table(mid, level="realistic", seed=1, force=False):
    path = os.path.join(TABLE_DIR, "{}_{}_s{}.parquet".format(mid, level, seed))
    if os.path.exists(path) and not force:
        return pd.read_parquet(path)
    m = parse.load_match(mid, quiet=True)
    dead = m["status"] == 0
    lv = LEVELS[level]
    d = degrade.degrade(to_atd.clean_arrays(m), hazards(), dead, seed=seed,
                        scale_loss=_scaled((0.5, 3.0), lv["loss"]),
                        scale_regain=_scaled((2.5, 0.2), lv["regain"]))
    df, _ = to_atd.build_frames(d)
    df["label"] = bench.variant_b_codes(mid, m).astype(np.int8)
    df["dead"] = dead
    df["match"], df["level"], df["seed"] = mid, level, seed
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def feature_columns(df):
    return [c for c in df.columns
            if c not in EXCLUDE and c not in LABEL_COLUMNS and not c.startswith(EXCLUDE_PREFIXES)
            and pd.api.types.is_numeric_dtype(df[c])]
