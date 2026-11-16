"""Experiments for a bigger jump (2026-11-16): where is the score lost, and what recovers it.

    python -m src.idsse.experiments oracle     # replace one part of the prediction with the truth
    python -m src.idsse.experiments decoding   # two-stage decoding, short-possession cleanup

All on the five DEV folds with the ensemble's leave-one-match-out probabilities (LightGBM 0.5, TCN 0.25,
GRU 0.25). Every setting is chosen per fold on its four training matches (boundaries.cv_pick) and
scored on its validation match. The oracle runs are diagnostics only (they use the validation labels).
"""

import itertools
import sys

import numpy as np
import pandas as pd

from . import bench, boundaries, dataset, decode, labels, splits

FPS = 25
H, A, D = labels.H, labels.A, labels.D
BIG = 50.0


def cases():
    """{match: dict(lp, half, gt, seen, p)} for the DEV matches."""
    out = {}
    for v, (lp, h, gt, contact) in boundaries._cases().items():
        seen = dataset.match_table(v, "realistic", 1)["ball_observed"].to_numpy(bool)
        out[v] = {"lp": lp, "half": h, "gt": gt, "seen": seen, "p": np.exp(lp), "contact": contact}
    return out


def loo_costs(cs):
    grid = decode.grid_scores({v: (c["lp"], c["half"], c["gt"]) for v, c in cs.items()})
    return {v: decode.pick_costs(grid, [m for m in cs if m != v]) for v in cs}


def force(lp, mask, codes):
    """Log-probabilities that force the given state on the masked frames."""
    out = lp.copy()
    out[mask] = -BIG
    out[mask, codes[mask]] = 0.0
    return out


def live_starts(codes, half):
    """First frame of every live stretch (after a dead ball or at a half's start)."""
    live = codes != D
    prev = np.r_[False, live[:-1]] & np.r_[False, half[1:] == half[:-1]]
    return np.flatnonzero(live & ~prev)


def turnover_frames(codes, half):
    return np.flatnonzero((codes[1:] != codes[:-1]) & (codes[1:] != D) & (codes[:-1] != D)
                          & (half[1:] == half[:-1])) + 1


# ---- 1. oracle: which part, if perfect, gains most -------------------------------------------------

def oracle():
    cs = cases()
    costs = loo_costs(cs)
    rows = []
    for v, c in cs.items():
        lp, h, gt = c["lp"], c["half"], c["gt"]
        live = gt != D
        masks = {"as is": np.zeros(len(gt), bool),
                 "perfect dead/live": np.ones(len(gt), bool)}
        st = live_starts(gt, h)
        restart = np.zeros(len(gt), bool)
        for s in st:
            restart[s:s + 2 * FPS] = live[s:s + 2 * FPS]
        masks["perfect team, first 2 s after restarts"] = restart
        masks["perfect team while ball unseen"] = live & ~c["seen"]
        masks["perfect team while ball seen"] = live & c["seen"]
        turn = np.zeros(len(gt), bool)
        for f in turnover_frames(gt, h):
            turn[max(f - FPS, 0):f + FPS] = live[max(f - FPS, 0):f + FPS]
        masks["perfect team +-1 s around turnovers"] = turn
        masks["perfect team, all live frames"] = live
        for name, m in masks.items():
            if name == "perfect dead/live":
                lpf = lp.copy()
                lpf[gt == D, :2] = -BIG
                lpf[gt == D, 2] = 0.0
                lpf[gt != D, 2] = -BIG
            else:
                lpf = force(lp, m, gt)
            codes = decode.viterbi(lpf, h, **costs[v])
            s = bench.score(gt, codes, h)
            rows.append({"match": v, "oracle": name, "spellF1": s["spellF1"], "macroF1": s["macroF1"],
                         "share_of_frames": m.mean()})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.groupby("oracle", sort=False)[["spellF1", "macroF1", "share_of_frames"]].mean().round(3).to_string())
    return df


# ---- 2. decoding variants ------------------------------------------------------------------------

def two_state(lp2, half, cost):
    """Viterbi over two states (columns of lp2) with one switch cost, per half."""
    lp3 = np.c_[lp2, np.full(len(lp2), -BIG)]
    return decode.viterbi(lp3, half, team_cost=cost, dead_cost=BIG * 10)


def two_stage(p, half, dead_cost, team_cost, min_dead=0):
    """Stage 1: dead / live. Stage 2: home / away inside each live stretch, from the team probabilities
    renormalised without dead ball. min_dead: stoppages shorter than this (frames) are removed."""
    pd_ = np.clip(p[:, D], 1e-6, 1 - 1e-6)
    dead = two_state(np.c_[np.log(1 - pd_), np.log(pd_)], half, dead_cost) == 1
    if min_dead:
        d = np.diff(np.r_[False, dead, False].astype(np.int8))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a < min_dead:
                dead[a:b] = False
    ph = np.clip(p[:, H] / np.clip(p[:, H] + p[:, A], 1e-9, None), 1e-6, 1 - 1e-6)
    seg = np.where(dead, -1, 0) + half * 10                       # live stretches decoded separately
    blk = np.cumsum(np.r_[True, seg[1:] != seg[:-1]])
    team = two_state(np.c_[np.log(ph), np.log(1 - ph)], blk, team_cost)
    return np.where(dead, D, team).astype(np.int8)


def cleanup(codes, half, tau_frames, before_dead):
    return labels.give_back(codes, half, tau_frames, before_dead) if tau_frames else codes


def decoding():
    cs = cases()
    costs = loo_costs(cs)
    rows = []
    for v, c in cs.items():
        p, h, gt, lp = c["p"], c["half"], c["gt"], c["lp"]
        base = decode.viterbi(lp, h, **costs[v])
        for tau, bd in itertools.product((0, 0.5, 1.0, 2.0, 3.0), (False, True)):
            if tau == 0 and bd:
                continue
            s = bench.score(gt, cleanup(base, h, int(tau * FPS), bd), h)
            rows.append({"match": v, "method": "one-stage + cleanup", "setting": ("1s", tau, bd),
                         "spellF1": s["spellF1"], "macroF1": s["macroF1"]})
        for dc, tc, md in itertools.product((8, 16, 32, 64), (8, 16, 32, 64, 128), (0, FPS)):
            s = bench.score(gt, two_stage(p, h, dc, tc, md), h)
            rows.append({"match": v, "method": "two-stage", "setting": ("2s", dc, tc, md),
                         "spellF1": s["spellF1"], "macroF1": s["macroF1"]})
        print("decoded", v, flush=True)
    df = pd.DataFrame(rows)
    out = []
    for method in df.method.unique():
        r = boundaries.cv_pick(df[df.method == method])
        out.append({"method": method, "cv_spellF1": r.spellF1.mean(),
                    "settings_picked": "; ".join(str(s) for s in r.setting)})
    base = df[df.setting == ("1s", 0, False)].spellF1.mean()
    print("one-stage, no cleanup (current): {:.3f}".format(base))
    print(pd.DataFrame(out).round(3).to_string(index=False))
    df.to_pickle("cache/idsse/exp_decoding.pkl")
    return df


# ---- 3. new inputs: LightGBM cross-validation on the current input version ----------------------

def lgbm_cv():
    """LightGBM (best tuned settings) on the current inputs (inputs.VERSION); saves
    cache/idsse/cvpred_lgbm_<version>.npz and prints the cross-validated spellF1."""
    import os
    import time
    from . import inputs, parse, rate_match, tune_trees
    t0 = time.time()
    preds, cs = {}, {}
    for train, val in splits.cv_folds():
        X, y = inputs.train_arrays(train, tune_trees.LEVELS, tune_trees.SEED, step=5)
        model = tune_trees.make_model("lgbm", rate_match.LGBM_BEST)
        model.fit(X, y)
        a = inputs.get(val, "realistic", tune_trees.SEED)
        preds[val] = tune_trees.predict_avg(model, a).astype(np.float32)
        cs[val] = (np.log(np.clip(preds[val], 1e-6, 1)), a["half"], a["y"])
        print("fold {} ({} inputs) done [{:.0f} s]".format(val, X.shape[1], time.time() - t0), flush=True)
    np.savez(os.path.join(parse.CACHE_DIR, "cvpred_lgbm_{}.npz".format(inputs.VERSION)), **preds)
    res = decode.cv_scores(decode.grid_scores(cs), splits.cv_folds())
    print(res.round(3).to_string(index=False))
    print("LightGBM inputs {}: spellF1 {:.3f} | macroF1 {:.3f}   (inputs v1: 0.681)".format(
        inputs.VERSION, res.spellF1.mean(), res.macroF1.mean()))


if __name__ == "__main__":
    {"oracle": oracle, "decoding": decoding, "lgbm": lgbm_cv}[sys.argv[1]]()
