"""Choose the smoother's team-switch cost per match so the decoded possession rate matches a target,
instead of reusing one cost picked on IDSSE.

Why: the cost that is best on IDSSE assumes the model is as confident as it is on IDSSE. On BAR-ATH the
model is less sure, so the same cost merges possessions (public window: 8-10 predicted vs 17). Matching
the rate needs no labels of the match being decoded: the target is the possession rate of the IDSSE
training labels (variant B, ~4.2 possessions per minute of live play).

    tc = rate_match.team_cost_for_rate(logp, half, target, dead_cost=128)

    python -m src.idsse.rate_match cv     # test on IDSSE folds: fixed vs rate-matched costs
"""

import os
import sys
import time

import numpy as np
import pandas as pd

from . import bench, decode, inputs, labels, parse, splits, tune_trees

FPS = 25
TEAM_GRID = (1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 180, 256)
LGBM_BEST = {"num_leaves": 31, "min_child_samples": 50, "colsample_bytree": 0.3, "subsample": 0.7,
             "reg_lambda": 1.0}           # best LightGBM settings from the tuning (docs/IDSSE_RESULTS.md)


def possession_rate(codes, half):
    """Possessions (runs of H or A) per minute of live play."""
    sp = labels.spells(codes, half)
    live_min = (codes != labels.D).sum() / FPS / 60
    return int((sp[:, 2] != labels.D).sum()) / max(live_min, 1e-9)


def team_cost_for_rate(logp, half, target, dead_cost, grid=TEAM_GRID):
    """The team-switch cost whose decoded possession rate is closest to `target`."""
    best = None
    for tc in grid:
        r = possession_rate(decode.viterbi(logp, half, tc, dead_cost), half)
        if best is None or abs(r - target) < best[1]:
            best = (tc, abs(r - target), r)
    return best[0], best[2]


def cv():
    t0 = time.time()
    folds = splits.cv_folds()
    preds = {}
    for train, val in folds:
        X, y = inputs.train_arrays(train, tune_trees.LEVELS, tune_trees.SEED, step=5)
        model = tune_trees.make_model("lgbm", LGBM_BEST)
        model.fit(X, y)
        del X, y
        for level in ("realistic", "heavy"):
            a = inputs.get(val, level, tune_trees.SEED)
            preds[(val, level)] = (np.log(np.clip(tune_trees.predict_avg(model, a), 1e-6, 1)), a["half"], a["y"])
        print("fold {} trained ({:.0f} s)".format(val, time.time() - t0), flush=True)
    np.savez(os.path.join(parse.CACHE_DIR, "cvpred_lgbm_local.npz"),
             **{"{}_{}".format(v, l): p[0] for (v, l), p in preds.items()})

    # fixed costs: picked per fold on the other matches (realistic level), as before
    grid = decode.grid_scores({v: preds[(v, "realistic")] for _, v in folds})
    rows = []
    for train, val in folds:
        fixed = decode.pick_costs(grid, train)
        target = np.mean([possession_rate(inputs.get(m, "realistic", tune_trees.SEED)["y"],
                                          inputs.get(m, "realistic", tune_trees.SEED)["half"]) for m in train])
        for level in ("realistic", "heavy"):
            lp, h, gt = preds[(val, level)]
            s_fixed = bench.score(gt, decode.viterbi(lp, h, **fixed), h)
            tc, rate = team_cost_for_rate(lp, h, target, fixed["dead_cost"])
            s_rate = bench.score(gt, decode.viterbi(lp, h, tc, fixed["dead_cost"]), h)
            rows.append({"validate": val, "level": level, "target_rate": target,
                         "fixed_cost": fixed["team_cost"], "fixed_spellF1": s_fixed["spellF1"],
                         "fixed_n": s_fixed["n_pred_spells"], "rate_cost": tc, "rate_spellF1": s_rate["spellF1"],
                         "rate_n": s_rate["n_pred_spells"], "gt_n": s_fixed["n_gt_spells"]})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(parse.CACHE_DIR, "rate_match_cv.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(res.groupby("level")[["fixed_spellF1", "rate_spellF1", "fixed_n", "rate_n", "gt_n"]].mean().round(3))


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        cv()
