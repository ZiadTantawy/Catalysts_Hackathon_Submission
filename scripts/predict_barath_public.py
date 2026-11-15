"""Train best IDSSE tree model(s) on all IDSSE matches, predict BAR-ATH, score public 5 min.

    BARATH_CACHE=... python scripts/predict_barath_public.py

Writes evaluation/results/idsse_*.csv and prints the public scoreboard.
Does not use the public GT for training or cost selection (costs from IDSSE DEV only).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.idsse import decode, inputs, labels, prep, splits, tune_trees
from src.idsse.degrade import barath_cache_dir

OUT_RESULTS = os.path.join(ROOT, "evaluation", "results")
SEED = tune_trees.SEED
LEVELS = list(tune_trees.LEVELS)


def barath_arrays(feats):
    """X, Xm, half, graded for BAR-ATH, aligned to the IDSSE feature list."""
    path = os.path.join(barath_cache_dir(), "frames.parquet")
    df = prep.add_long_context(pd.read_parquet(path))
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise RuntimeError("BAR-ATH missing {} features, e.g. {}".format(len(missing), missing[:5]))
    sw = prep.swap_home_away(df)
    return {
        "X": df[feats].to_numpy(np.float32),
        "Xm": sw[feats].to_numpy(np.float32),
        "half": df["half"].to_numpy(np.int8),
        "graded": df["graded"].to_numpy(bool),
        "frame": df["frame"].to_numpy(np.int32),
    }


def train_full(kind, params, step=5):
    print("training {} on all IDSSE (step={}) ...".format(kind, step), flush=True)
    t0 = time.time()
    X, y = inputs.train_arrays(splits.ALL, LEVELS, SEED, step)
    model = tune_trees.make_model(kind, params)
    model.fit(X, y)
    print("  fit {:,} rows in {:.0f} s".format(len(y), time.time() - t0), flush=True)
    return model


def predict_codes(model, a, team_cost, dead_cost):
    p = tune_trees.predict_avg(model, a)
    logp = np.log(np.clip(p, 1e-6, 1))
    codes = decode.viterbi(logp, a["half"], team_cost=team_cost, dead_cost=dead_cost)
    # Ungraded frames -> X (submission convention used by baselines for off-window frames)
    letters = np.where(a["graded"], labels.LETTERS[codes], "X")
    return letters, p


def pick_costs_from_cv(kind):
    """Decoder costs chosen only on IDSSE DEV (never on BAR-ATH / public GT)."""
    preds = np.load(os.path.join(tune_trees.OUT, "cvpred_{}.npz".format(kind)))
    cases = {}
    for _, val in splits.cv_folds():
        a = inputs.get(val, "realistic", SEED)
        cases[val] = (np.log(np.clip(preds[val], 1e-6, 1)), a["half"], a["y"])
    grid = decode.grid_scores(cases)
    return decode.pick_costs(grid, splits.DEV)


def write_submission(path, frames, letters):
    pd.DataFrame({"frame": frames, "label": letters}).to_csv(path, index=False)
    print("wrote", path, flush=True)


def main():
    os.makedirs(OUT_RESULTS, exist_ok=True)
    feats = list(inputs.get(splits.DEV[0], "realistic", SEED)["feats"])
    print("features:", len(feats), flush=True)
    bar = barath_arrays(feats)
    print("BAR-ATH frames:", len(bar["frame"]), "graded:", int(bar["graded"].sum()), flush=True)

    kinds = ["lgbm", "xgb", "rf"]
    probs = {}
    costs = {}
    for kind in kinds:
        params = tune_trees.best_params(kind)
        costs[kind] = pick_costs_from_cv(kind)
        print(kind, "params", params, "costs", costs[kind], flush=True)
        model = train_full(kind, params, step=5)
        letters, p = predict_codes(model, bar, **costs[kind])
        probs[kind] = p
        out = os.path.join(OUT_RESULTS, "idsse_{}.csv".format(kind))
        write_submission(out, bar["frame"], letters)

    # Average of the three probability heads, decode with lgbm's IDSSE costs
    p_avg = np.mean([probs[k] for k in kinds], 0)
    logp = np.log(np.clip(p_avg, 1e-6, 1))
    codes = decode.viterbi(logp, bar["half"], **costs["lgbm"])
    letters = np.where(bar["graded"], labels.LETTERS[codes], "X")
    write_submission(os.path.join(OUT_RESULTS, "idsse_combined.csv"), bar["frame"], letters)

    print("\n=== public 5-minute scoreboard ===", flush=True)
    os.chdir(ROOT)
    os.system("python3 evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
