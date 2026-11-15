"""Tune random forest, LightGBM and XGBoost on the DEV folds; then score the best of each, and their
combination, with the full possession score.

    python -m src.idsse.tune_trees tune xgb      # random search; appends to cache/idsse/tuning_xgb.csv
    python -m src.idsse.tune_trees tune lgbm
    python -m src.idsse.tune_trees tune rf
    python -m src.idsse.tune_trees final         # best config of each -> fold predictions -> spellF1

Tuning score: mean log-loss on the validation match (realistic level, prediction averaged over the
frame and its home/away mirror), over the five leave-one-match-out folds. Cheap and smooth, so many
settings can be compared; the possession score (spellF1, after the smoother) is computed for the
winners only. Tuning trains on every 10th frame; the final fits on every 5th. Test matches: never.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import dataset, decode, inputs, parse, splits

LEVELS = list(dataset.LEVELS)
SEED = 1
OUT = parse.CACHE_DIR
RNG = np.random.default_rng(20261116)
N_JOBS = os.cpu_count() or 4


def _xgb_device():
    """CUDA when present; CPU on Apple Silicon / machines without an NVIDIA GPU."""
    try:
        import xgboost as xgb
        build = xgb.build_info() if hasattr(xgb, "build_info") else {}
        if build.get("USE_CUDA"):
            return "cuda"
    except Exception:
        pass
    return "cpu"

SPACES = {
    "lgbm": {"num_leaves": [31, 63, 127, 255], "min_child_samples": [50, 200, 800],
             "colsample_bytree": [0.3, 0.6, 0.9], "subsample": [0.7, 1.0], "reg_lambda": [0.0, 1.0, 10.0]},
    "xgb": {"max_depth": [4, 6, 8, 10], "min_child_weight": [1, 10, 50], "colsample_bytree": [0.3, 0.6, 0.9],
            "subsample": [0.7, 1.0], "reg_lambda": [1.0, 10.0]},
    "rf": {"max_features": [0.05, 0.1, 0.2], "min_samples_leaf": [20, 100, 400]},
}
N_TRIALS = {"lgbm": 10, "xgb": 12, "rf": 6}


def make_model(kind, params, n_trees=None):
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="multiclass", n_estimators=n_trees or 300, learning_rate=0.1,
                                  subsample_freq=1, verbose=-1, random_state=0, n_jobs=N_JOBS, **params)
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(objective="multi:softprob", n_estimators=n_trees or 300, learning_rate=0.1,
                                 tree_method="hist", device=_xgb_device(), random_state=0,
                                 n_jobs=N_JOBS, **params)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=n_trees or 200, max_samples=0.3, n_jobs=N_JOBS,
                                      random_state=0, **params)
    raise ValueError(kind)


def predict_avg(model, a):
    """P(H, A, D) averaged over the frames and their mirror (mirror's H and A exchanged back)."""
    p = model.predict_proba(a["X"])
    pm = model.predict_proba(a["Xm"])[:, [1, 0, 2]]
    return (p + pm) / 2


def logloss(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1))))


def run_folds(kind, params, step, n_trees=None, keep=False):
    """Train per fold, predict the validation match. Returns mean log-loss (and the predictions)."""
    losses, preds = [], {}
    for train, val in splits.cv_folds():
        X, y = inputs.train_arrays(train, LEVELS, SEED, step)
        model = make_model(kind, params, n_trees)
        model.fit(X, y)
        del X, y
        a = inputs.get(val, "realistic", SEED)
        p = predict_avg(model, a)
        losses.append(logloss(p, a["y"]))
        if keep:
            preds[val] = p.astype(np.float32)
    return float(np.mean(losses)), losses, preds


def sample(kind):
    return {k: v[RNG.integers(len(v))] for k, v in SPACES[kind].items()}


def tune(kind):
    path = os.path.join(OUT, "tuning_{}.csv".format(kind))
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    trials = [{k: v[len(v) // 2] for k, v in SPACES[kind].items()}]      # middle settings first
    trials += [sample(kind) for _ in range(N_TRIALS[kind] - 1)]
    for k, params in enumerate(trials, 1):
        key = json.dumps(params, sort_keys=True, default=float)
        if len(done) and key in set(done["params"]):
            continue
        t = time.time()
        mean, per, _ = run_folds(kind, params, step=10)
        row = pd.DataFrame([{"params": key, "logloss": mean, "per_fold": json.dumps([round(x, 4) for x in per]),
                             "seconds": round(time.time() - t)}])
        row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print("[{} {}/{}] logloss {:.4f} ({:.0f} s) {}".format(kind, k, len(trials), mean, time.time() - t, key),
              flush=True)


def best_params(kind):
    df = pd.read_csv(os.path.join(OUT, "tuning_{}.csv".format(kind)))
    return json.loads(df.sort_values("logloss").iloc[0]["params"])


def final():
    """Best config of each model, fit on every 5th frame; fold predictions saved; spellF1 per model and
    for the average of the three."""
    folds = splits.cv_folds()
    halves = {v: inputs.get(v, "realistic", SEED)["half"] for _, v in folds}
    labels = {v: inputs.get(v, "realistic", SEED)["y"] for _, v in folds}
    probs, rows = {}, []
    for kind in ("rf", "lgbm", "xgb"):
        params = best_params(kind)
        t = time.time()
        mean, per, preds = run_folds(kind, params, step=5, keep=True)
        probs[kind] = preds
        np.savez(os.path.join(OUT, "cvpred_{}.npz".format(kind)), **preds)
        print("{} best {} | logloss {:.4f} ({:.0f} s)".format(kind, params, mean, time.time() - t), flush=True)
    probs["combined"] = {v: np.mean([probs[k][v] for k in ("rf", "lgbm", "xgb")], 0) for _, v in folds}
    for name, pr in probs.items():
        cases = {v: (np.log(np.clip(pr[v], 1e-6, 1)), halves[v], labels[v]) for _, v in folds}
        cv = decode.cv_scores(decode.grid_scores(cases), folds)
        live = {v: labels[v] != 2 for _, v in folds}
        team = np.mean([np.mean(pr[v].argmax(1)[live[v]] == labels[v][live[v]]) for _, v in folds])
        rows.append({"model": name, "spellF1": cv.spellF1.mean(), "macroF1": cv.macroF1.mean(),
                     "team_right_live": team, "logloss": np.mean([logloss(pr[v], labels[v]) for _, v in folds]),
                     **{"spell_" + v: s for v, s in zip(cv.validate, cv.spellF1)}})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "cv_trees_final.csv"), index=False)
    print(res.round(3).to_string(index=False))
    print("reference: simple evidence + smoother 0.343 | organisers' majority vote 0.208")


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune(sys.argv[2])
    else:
        final()
