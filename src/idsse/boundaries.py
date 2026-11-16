"""Sharper possession changes (improvement 2).

2a  snap_to_contact: move each decoded change of team to the new team's nearest touch (the ETL's
    `contact_team`: a measured player close to a ball that reacts). The scorer charges a change that is
    one second late heavily (EVALUATION.md: 1 s late -> spellF1 0.65), and the ground truth flips near
    the new team's first contact (docs/PLAN.md).

    python -m src.idsse.boundaries snap      # cross-validated test on the IDSSE folds
"""

import sys

import numpy as np
import pandas as pd

from . import dataset, decode, ensemble, inputs, labels, splits

FPS = 25
ENSEMBLE = {"lgbm_local": 0.5, "nn_tcn": 0.25, "nn_gru": 0.25}


def snap_to_contact(codes, contact_sign, half, window):
    """Move every live H <-> A change to the start of the new team's touch nearest to it, within
    +-window frames. contact_sign: +1 home touch, -1 away touch, 0 none (ETL convention)."""
    out = codes.copy()
    n = len(codes)
    ch = np.flatnonzero((codes[1:] != codes[:-1]) & (codes[1:] != labels.D) & (codes[:-1] != labels.D)
                        & (half[1:] == half[:-1])) + 1
    for f in ch:
        new = codes[f]
        sign = 1 if new == labels.H else -1
        lo, hi = max(f - window, 1), min(f + window, n - 1)
        touch = np.flatnonzero(contact_sign[lo:hi + 1] == sign) + lo
        if not len(touch):
            continue
        starts = touch[np.r_[True, np.diff(touch) > 1]]           # first frame of each touch
        g = starts[np.argmin(np.abs(starts - f))]
        a, b = sorted((f, g))
        seg = out[a:b]
        if (seg == labels.D).any() or (half[a:b + 1] != half[f]).any():
            continue                                              # never move a change across dead ball
        out[a:b] = new if g < f else codes[f - 1]
    return out


def _cases():
    cv = {k: ensemble.load(k) for k in ENSEMBLE}
    out = {}
    for v in splits.DEV:
        a = inputs.get(v, "realistic", 1)
        lp = np.log(np.clip(sum(w * cv[k][v] for k, w in ENSEMBLE.items()), 1e-6, 1))
        contact = dataset.match_table(v, "realistic", 1)["contact_team"].to_numpy()
        out[v] = (lp, a["half"], a["y"], contact)
    return out


def loo_decoded(cases):
    """Each match decoded with smoother costs picked on the other DEV matches (its fold's costs)."""
    grid = decode.grid_scores({v: c[:3] for v, c in cases.items()})
    return {v: decode.viterbi(cases[v][0], cases[v][1], **decode.pick_costs(grid, [m for m in cases if m != v]))
            for v in cases}


def cv_pick(scores):
    """scores: DataFrame(match, setting, spellF1). Per fold: setting picked on its training matches,
    scored on its validation match. Returns per-fold rows."""
    res = []
    for train, val in splits.cv_folds():
        best = scores[scores.match.isin(train)].groupby("setting").spellF1.mean().idxmax()
        res.append({"validate": val, "setting": best,
                    "spellF1": scores[(scores.match == val) & (scores.setting == best)].spellF1.iloc[0]})
    return pd.DataFrame(res)


def test_snap(windows_s=(0, 0.25, 0.5, 1.0, 2.0)):
    from . import bench
    cases = _cases()
    decoded = loo_decoded(cases)
    rows = []
    for v, (lp, h, gt, contact) in cases.items():
        for w in windows_s:
            codes = snap_to_contact(decoded[v], contact, h, int(w * FPS)) if w else decoded[v]
            rows.append({"match": v, "setting": w, "spellF1": bench.score(gt, codes, h)["spellF1"]})
    scores = pd.DataFrame(rows)
    print(scores.groupby("setting").spellF1.mean().round(3).rename("mean spellF1 by snap window (s)").to_string())
    res = cv_pick(scores)
    base = scores[scores.setting == 0].set_index("match").spellF1
    res["no_snap"] = res.validate.map(base)
    print(res.round(3).to_string(index=False))
    print("cross-validated: no snap {:.3f} -> snap {:.3f}".format(res.no_snap.mean(), res.spellF1.mean()))


# ---- 2b: a turnover model and a smoother whose switch cost follows it ------------------------------
#
# Diagnosis on the IDSSE folds (2026-11-16, ensemble + smoother): only 61% of live H <-> A changes are
# found within 3 s (restarts 81%, ball going out 86%), 29% of predicted ones are not real, and found
# ones are typically 0.7 s off. A dedicated model answers "does possession change within +-0.5 s?";
# the smoother's H <-> A cost becomes cheap where it says yes:  cost_t = c0 * exp(-beta * z_t),
# z_t = logit(p_t) - median logit. beta = 0 is the plain smoother.

TURN_RADIUS = 12                     # frames (0.5 s) either side of a change count as positive
TURN_PARAMS = {"num_leaves": 31, "min_child_samples": 50, "colsample_bytree": 0.3, "subsample": 0.7,
               "reg_lambda": 1.0}


def turnover_target(y, half):
    ch = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] != labels.D) & (y[:-1] != labels.D)
                        & (half[1:] == half[:-1])) + 1
    t = np.zeros(len(y), np.int8)
    for f in ch:
        t[max(f - TURN_RADIUS, 0):f + TURN_RADIUS] = 1
    return t


def turnover_cv(step=5):
    """Leave-one-match-out turnover probabilities for the DEV matches (cached)."""
    import os
    import lightgbm as lgb
    from . import parse, tune_trees
    path = os.path.join(parse.CACHE_DIR, "cvpred_turnover.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {v: z[v] for v in z.files}
    preds = {}
    for train, val in splits.cv_folds():
        X, t = [], []
        for m in train:
            for level in tune_trees.LEVELS:
                a = inputs.get(m, level, 1)
                tt = turnover_target(a["y"], a["half"])[::step]
                X += [a["X"][::step], a["Xm"][::step]]
                t += [tt, tt]                                      # a turnover is one in the mirror too
        clf = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1, subsample_freq=1,
                                 verbose=-1, random_state=0, n_jobs=tune_trees.N_JOBS, **TURN_PARAMS)
        clf.fit(np.concatenate(X), np.concatenate(t))
        a = inputs.get(val, "realistic", 1)
        preds[val] = ((clf.predict_proba(a["X"])[:, 1] + clf.predict_proba(a["Xm"])[:, 1]) / 2).astype(np.float32)
        print("turnover model fold {} done".format(val), flush=True)
    np.savez(path, **preds)
    return preds


def team_cost_series(p_turn, c0, beta):
    lg = np.log(np.clip(p_turn, 1e-4, 1 - 1e-4) / np.clip(1 - p_turn, 1e-4, 1))
    return c0 * np.exp(-beta * np.clip(lg - np.median(lg), -6, 6))


def test_turnover(c0s=(8, 16, 32, 64, 128), betas=(0.0, 0.3, 0.6, 1.0, 1.5)):
    from . import bench
    cases = _cases()
    pt = turnover_cv()
    from sklearn.metrics import roc_auc_score
    for v in cases:
        t = turnover_target(cases[v][2], cases[v][1])
        print("turnover model {}: AUC {:.3f}".format(v, roc_auc_score(t, pt[v])))
    grid = decode.grid_scores({v: c[:3] for v, c in cases.items()})
    rows = []
    for v, (lp, h, gt, _) in cases.items():
        dead_cost = decode.pick_costs(grid, [m for m in cases if m != v])["dead_cost"]
        for c0 in c0s:
            for beta in betas:
                codes = decode.viterbi(lp, h, c0, dead_cost, team_cost_t=team_cost_series(pt[v], c0, beta))
                rows.append({"match": v, "setting": (c0, beta), "spellF1": bench.score(gt, codes, h)["spellF1"]})
    scores = pd.DataFrame(rows)
    plain = scores[scores.setting.map(lambda s: s[1] == 0.0)]
    res_plain = cv_pick(plain)
    res = cv_pick(scores)
    res["plain_smoother"] = res_plain.spellF1.to_numpy()
    print(res.round(3).to_string(index=False))
    print("cross-validated: plain smoother {:.3f} -> turnover-aware {:.3f}".format(
        res.plain_smoother.mean(), res.spellF1.mean()))


if __name__ == "__main__":
    if sys.argv[1] == "snap":
        test_snap()
    elif sys.argv[1] == "turnover":
        test_turnover()
