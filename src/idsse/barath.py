"""Apply an IDSSE-trained model to BAR-ATH, with the smoother's team cost calibrated without BAR-ATH labels.

    python -m src.idsse.barath            # LightGBM on all 7 IDSSE matches -> BAR-ATH, decoded twice
    python -m src.idsse.barath --retrain  # ignore the cached BAR-ATH probabilities

Calibration (src/idsse/rate_match.py, tested 2026-11-16): on the IDSSE folds the best fixed cost decodes
~91% of the true number of possessions, and matching the true count exactly scores slightly worse. So
BAR-ATH is decoded with the cost whose possession rate equals the rate that the fixed cost produces on
the IDSSE validation matches: the same behaviour as where it could be checked, with no BAR-ATH labels.

Published fact applied after decoding: the graded frames before the second-half kickoff are dead
(FAQ.md; config.HALF_BLOCKS / KICKOFF_2ND). Ungraded frames are written as X.
Outputs: evaluation/results/idsse_lgbm_fixed.csv and idsse_lgbm_rated.csv, cache/idsse/barath_probs_lgbm.npy.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

from ..etl import config as C
from . import decode, degrade, inputs, labels, parse, prep, rate_match, splits, tune_trees

RESULTS = os.path.join(parse.REPO_ROOT, "evaluation", "results")
FPS = 25


def arrays(feats):
    """BAR-ATH inputs in the IDSSE feature order (frames table from the etl-v1.3 build)."""
    df = prep.add_long_context(pd.read_parquet(os.path.join(degrade.barath_cache_dir(), "frames.parquet")))
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise RuntimeError("BAR-ATH lacks {} inputs, e.g. {}".format(len(missing), missing[:5]))
    sw = prep.swap_home_away(df, label_col=None)
    graded = df["graded"].to_numpy(bool)
    # decode each graded half as one block; everything else is a separate (ignored) block
    block = np.where(graded, df["half"].to_numpy(np.int8), 0)
    block = np.where(~graded, -np.cumsum(np.r_[True, graded[1:] != graded[:-1]]), block)
    known_dead = np.zeros(len(df), bool)
    known_dead[C.HALF_BLOCKS[1][1]:C.KICKOFF_2ND] = True
    return {"X": df[feats].to_numpy(np.float32), "Xm": sw[feats].to_numpy(np.float32),
            "block": block, "graded": graded, "known_dead": known_dead}


def idsse_target_rate(cv_logp, fixed):
    """Possession rate the fixed costs produce on the IDSSE validation matches."""
    rates = []
    for val, lp in cv_logp.items():
        a = inputs.get(val, "realistic", tune_trees.SEED)
        rates.append(rate_match.possession_rate(decode.viterbi(lp, a["half"], **fixed), a["half"]))
    return float(np.mean(rates))


def summary(codes, graded, name):
    c, g = codes[graded], np.ones(graded.sum(), np.int8)
    sp = labels.spells(c, g)
    pos = sp[sp[:, 2] != labels.D]
    dead = sp[sp[:, 2] == labels.D]
    live_min = (c != labels.D).sum() / FPS / 60
    return {"decode": name, "possessions": len(pos), "per_live_min": round(len(pos) / live_min, 2),
            "mean_possession_s": round(pos[:, 4].mean() / FPS, 1), "ball_in_play_min": round(live_min, 1),
            "stoppages": len(dead)}


def write(path, codes, graded):
    letters = np.where(graded, labels.LETTERS[codes], "X")
    pd.DataFrame({"frame": np.arange(len(codes)), "label": letters}).to_csv(path, index=False)


def main():
    t0 = time.time()
    cvz = np.load(os.path.join(parse.CACHE_DIR, "cvpred_lgbm_local.npz"))
    cv_logp = {v: cvz["{}_realistic".format(v)] for v in splits.DEV}
    cases = {v: (cv_logp[v], inputs.get(v, "realistic", 1)["half"], inputs.get(v, "realistic", 1)["y"])
             for v in splits.DEV}
    fixed = decode.pick_costs(decode.grid_scores(cases), splits.DEV)
    target = idsse_target_rate(cv_logp, fixed)
    true_rate = np.mean([rate_match.possession_rate(c[2], c[1]) for c in cases.values()])
    print("fixed costs {} | IDSSE rate with them {:.2f}/min (true {:.2f}) [{:.0f} s]".format(
        fixed, target, true_rate, time.time() - t0), flush=True)

    feats = list(inputs.get(splits.DEV[0], "realistic", 1)["feats"])
    bar = arrays(feats)
    probs_path = os.path.join(parse.CACHE_DIR, "barath_probs_lgbm.npy")
    if os.path.exists(probs_path) and "--retrain" not in sys.argv:
        p = np.load(probs_path)
    else:
        X, y = inputs.train_arrays(splits.ALL, tune_trees.LEVELS, tune_trees.SEED, step=5)
        model = tune_trees.make_model("lgbm", rate_match.LGBM_BEST)
        model.fit(X, y)
        del X, y
        p = tune_trees.predict_avg(model, bar)
        np.save(probs_path, p.astype(np.float32))
    logp = np.log(np.clip(p, 1e-6, 1))
    logp[bar["known_dead"]] = np.log([1e-6, 1e-6, 1.0])       # published fact, before decoding and calibrating
    print("trained on all 7 and predicted BAR-ATH [{:.0f} s]".format(time.time() - t0), flush=True)

    g = bar["graded"]
    rows = []
    blk = bar["block"]
    codes_fixed = decode.viterbi(logp, blk, **fixed)
    tc, _ = rate_match.team_cost_for_rate(logp[g], blk[g], target, fixed["dead_cost"])
    codes_rated = decode.viterbi(logp, blk, tc, fixed["dead_cost"])
    os.makedirs(RESULTS, exist_ok=True)
    for name, codes in (("fixed (team {})".format(fixed["team_cost"]), codes_fixed),
                        ("rate-calibrated (team {})".format(tc), codes_rated)):
        codes = codes.copy()
        codes[bar["known_dead"]] = labels.D
        rows.append(summary(codes, g, name))
        write(os.path.join(RESULTS, "idsse_lgbm_{}.csv".format("fixed" if name.startswith("fixed") else "rated")),
              codes, g)
    print(pd.DataFrame(rows).to_string(index=False))
    print("target rate {:.2f}/min (fixed-cost behaviour on IDSSE); football priors: ball in play ~52-58 min, "
          "~100-130 stoppages".format(target))


ENSEMBLE = {"lgbm_local": 0.5, "nn_tcn": 0.25, "nn_gru": 0.25}   # src/idsse/ensemble.py, 2026-11-16


def nn_probs(arch, bar):
    """Network trained on all seven IDSSE matches, BAR-ATH probabilities (cached)."""
    import torch
    from . import train_nn
    path = os.path.join(parse.CACHE_DIR, "barath_probs_nn_{}.npy".format(arch))
    if os.path.exists(path) and "--retrain" not in sys.argv:
        return np.load(path)
    torch.manual_seed(0)
    net, scaler = train_nn.train_fold(splits.ALL, np.random.default_rng(0), arch)
    p = train_nn.predict_match(net, scaler, {"X": bar["X"], "Xm": bar["Xm"], "half": bar["block"],
                                             "y": np.zeros(len(bar["X"]), np.int8)})
    np.save(path, p.astype(np.float32))
    return p


def main_ensemble():
    """The tree + network mix on BAR-ATH, smoother calibrated the same label-free way as main()."""
    from . import ensemble
    t0 = time.time()
    cv = {k: ensemble.load(k) for k in ENSEMBLE}
    cv_logp = {v: np.log(np.clip(sum(w * cv[k][v] for k, w in ENSEMBLE.items()), 1e-6, 1)) for v in splits.DEV}
    cases = {v: (cv_logp[v], inputs.get(v, "realistic", 1)["half"], inputs.get(v, "realistic", 1)["y"])
             for v in splits.DEV}
    fixed = decode.pick_costs(decode.grid_scores(cases), splits.DEV)
    target = idsse_target_rate(cv_logp, fixed)
    print("ensemble fixed costs {} | IDSSE rate with them {:.2f}/min [{:.0f} s]".format(
        fixed, target, time.time() - t0), flush=True)

    feats = list(inputs.get(splits.DEV[0], "realistic", 1)["feats"])
    bar = arrays(feats)
    probs = {"lgbm_local": np.load(os.path.join(parse.CACHE_DIR, "barath_probs_lgbm.npy"))}
    for arch in ("tcn", "gru"):
        probs["nn_" + arch] = nn_probs(arch, bar)
        print("{} trained on all 7 and predicted BAR-ATH [{:.0f} s]".format(arch, time.time() - t0), flush=True)
    p = sum(w * probs[k] for k, w in ENSEMBLE.items())
    logp = np.log(np.clip(p, 1e-6, 1))
    logp[bar["known_dead"]] = np.log([1e-6, 1e-6, 1.0])
    g = bar["graded"]
    tc, _ = rate_match.team_cost_for_rate(logp[g], bar["block"][g], target, fixed["dead_cost"])
    codes = decode.viterbi(logp, bar["block"], tc, fixed["dead_cost"])
    write(os.path.join(RESULTS, "idsse_ensemble_rated.csv"), codes, g)
    print(pd.DataFrame([summary(codes, g, "ensemble, rate-calibrated (team {})".format(tc))]).to_string(index=False))


if __name__ == "__main__":
    if "ensemble" in sys.argv:
        main_ensemble()
    else:
        main()
