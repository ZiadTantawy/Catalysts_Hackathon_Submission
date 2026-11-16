"""The final IDSSE-trained submission: train once, save the models, then build Catalysts.csv from them.

    python -m src.idsse.final predict            # models/idsse/ + BAR-ATH ETL cache -> Catalysts.csv
    python -m src.idsse.final train              # (re)train and save the models (needs IDSSE, see below)

Configuration (chosen on the IDSSE folds, docs/IDSSE_RESULTS.md):
- inputs v1: the etl-v1.3 frames table + 5 s / 15 s context (src/idsse/prep.py), no Smart Tagging,
  no camera columns, no ball-stillness flags;
- labels: variant B (possession changes only where the DFL event log confirms control);
- models: LightGBM 0.5 + TCN 0.25 + GRU 0.25, each averaged over the frame and its home/away mirror;
- smoother: Viterbi over H / A / D. The dead cost is the one picked on the IDSSE folds; the team cost is
  set on BAR-ATH so that the decoded possession rate equals the rate the IDSSE-picked costs give on the
  IDSSE validation matches (no BAR-ATH labels used);
- published fact: graded frames before the second-half kickoff are dead (FAQ.md); ungraded frames X.

Reproducibility: `predict` only reads the saved models (models/idsse/) and the BAR-ATH ETL cache, and runs
the networks on the CPU in float32, so the same inputs give the same file on any machine. `train` is
not bit-for-bit repeatable on a GPU, which is why its output is committed.

`train` needs the IDSSE data (python -c "from src.idsse import download; download.ensure()"), the
training tables (python -m src.idsse.build_tables), the BAR-ATH ETL cache (for the degrader's ball
model) and the cross-validation predictions cvpred_lgbm_local / cvpred_nn_tcn / cvpred_nn_gru
(python -m src.idsse.rate_match cv; python -m src.idsse.train_nn cv tcn; ... cv gru) for the
smoother settings.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import barath, decode, inputs, labels, parse, rate_match, splits

MODEL_DIR = os.path.join(parse.REPO_ROOT, "models", "idsse")
OUT_DEFAULT = os.path.join(parse.REPO_ROOT, "Catalysts.csv")
WEIGHTS = {"lgbm": 0.5, "tcn": 0.25, "gru": 0.25}
CV_NAMES = {"lgbm": "lgbm_local", "tcn": "nn_tcn", "gru": "nn_gru"}


# ---- train -----------------------------------------------------------------------------------------

def train():
    import lightgbm as lgb
    import torch
    from . import ensemble, train_nn, tune_trees
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    feats = list(inputs.get(splits.DEV[0], "realistic", 1)["feats"])

    # smoother settings from the cross-validation predictions of the same three models
    cv = {k: ensemble.load(CV_NAMES[k]) for k in WEIGHTS}
    cv_logp = {v: np.log(np.clip(sum(w * cv[k][v] for k, w in WEIGHTS.items()), 1e-6, 1)) for v in splits.DEV}
    cases = {v: (cv_logp[v], inputs.get(v, "realistic", 1)["half"], inputs.get(v, "realistic", 1)["y"])
             for v in splits.DEV}
    fixed = decode.pick_costs(decode.grid_scores(cases), splits.DEV)
    target = barath.idsse_target_rate(cv_logp, fixed)
    meta = {"feats": feats, "weights": WEIGHTS, "dead_cost": float(fixed["dead_cost"]),
            "idsse_team_cost": float(fixed["team_cost"]), "target_rate_per_live_min": target,
            "team_cost_grid": list(rate_match.TEAM_GRID), "inputs_version": inputs.VERSION,
            "labels": inputs.LABELS, "trained_on": splits.ALL}
    print("smoother: dead cost {} | target {:.3f} possessions per live minute [{:.0f} s]".format(
        meta["dead_cost"], target, time.time() - t0), flush=True)

    X, y = inputs.train_arrays(splits.ALL, tune_trees.LEVELS, tune_trees.SEED, step=5)
    model = lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.1, subsample_freq=1,
                               verbose=-1, random_state=0, n_jobs=tune_trees.N_JOBS, deterministic=True,
                               force_col_wise=True, **rate_match.LGBM_BEST)
    model.fit(X, y)
    del X, y
    model.booster_.save_model(os.path.join(MODEL_DIR, "lgbm.txt"))
    print("LightGBM saved [{:.0f} s]".format(time.time() - t0), flush=True)

    for arch in ("tcn", "gru"):
        torch.manual_seed(0)
        net, scaler = train_nn.train_fold(splits.ALL, np.random.default_rng(0), arch)
        torch.save({"arch": arch, "n_in": 2 * len(feats), "state": {k: v.float().cpu() for k, v in net.state_dict().items()},
                    "med": scaler.med, "scale": scaler.scale}, os.path.join(MODEL_DIR, "nn_{}.pt".format(arch)))
        print("{} saved [{:.0f} s]".format(arch, time.time() - t0), flush=True)
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


# ---- predict ---------------------------------------------------------------------------------------

def _nn_probs(path, bar):
    """Saved network on BAR-ATH, on the CPU in float32 (same result on every machine)."""
    import torch
    from . import train_nn
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = train_nn.make_net(ck["arch"], ck["n_in"])
    net.load_state_dict(ck["state"])
    net.eval()
    scaler = train_nn.Scaler()
    scaler.med, scaler.scale = ck["med"], ck["scale"]
    torch.set_num_threads(1)                                   # fixed summation order
    p = np.zeros((len(bar["X"]), 3), np.float32)
    with torch.no_grad():
        for b in np.unique(bar["block"]):
            idx = np.flatnonzero(bar["block"] == b)
            st = idx[::train_nn.STEP]
            out = []
            for key, swap in (("X", False), ("Xm", True)):
                z = torch.from_numpy(scaler.transform(bar[key][st])[None])
                q = torch.softmax(net(z)[0].float(), -1).numpy()
                out.append(q[:, [1, 0, 2]] if swap else q)
            p[idx] = np.repeat((out[0] + out[1]) / 2, train_nn.STEP, 0)[:len(idx)]
    return p


def _lgbm_probs(path, bar):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=path)
    p = booster.predict(bar["X"], num_threads=1)
    pm = booster.predict(bar["Xm"], num_threads=1)[:, [1, 0, 2]]
    return (p + pm) / 2


def predict(out=OUT_DEFAULT):
    t0 = time.time()
    with open(os.path.join(MODEL_DIR, "meta.json")) as fh:
        meta = json.load(fh)
    bar = barath.arrays(meta["feats"])
    probs = {"lgbm": _lgbm_probs(os.path.join(MODEL_DIR, "lgbm.txt"), bar)}
    for arch in ("tcn", "gru"):
        probs[arch] = _nn_probs(os.path.join(MODEL_DIR, "nn_{}.pt".format(arch)), bar)
    print("models applied to BAR-ATH [{:.0f} s]".format(time.time() - t0), flush=True)
    p = sum(w * probs[k] for k, w in meta["weights"].items())
    logp = np.log(np.clip(p, 1e-6, 1)).astype(np.float64)
    logp[bar["known_dead"]] = np.log([1e-6, 1e-6, 1.0])
    g = bar["graded"]
    tc, rate = rate_match.team_cost_for_rate(logp[g], bar["block"][g], meta["target_rate_per_live_min"],
                                             meta["dead_cost"], grid=meta["team_cost_grid"])
    codes = decode.viterbi(logp, bar["block"], tc, meta["dead_cost"])
    barath.write(out, codes, g)
    s = barath.summary(codes, g, "final")
    print("team cost {} (rate {:.2f}/min, target {:.2f}) | {} possessions, mean {} s, ball in play {} min, "
          "{} stoppages".format(tc, rate, meta["target_rate_per_live_min"], s["possessions"],
                                s["mean_possession_s"], s["ball_in_play_min"], s["stoppages"]))
    with open(out, "rb") as fh:
        print("wrote {} | sha256 {}".format(out, hashlib.sha256(fh.read()).hexdigest()))


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        predict(sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT)
