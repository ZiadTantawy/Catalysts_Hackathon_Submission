"""Improvement 3: the team's dead-ball detector (src/model/deadball.py) as extra evidence for D.

    python -m src.idsse.deadball_fusion cv      # run the detector on the DEV matches, test the fusion

The detector's rules were set on BAR-ATH (and checked on its video, docs/PLAN.md). On IDSSE it runs
without Smart Tagging, so its goal and kickoff-tag cues are silent there: the IDSSE test can only
undersell it. Fusion, before the smoother:

    log P(D) += alpha * evidence     evidence = "decision": +1 where the detector says dead, -1 live
                                     evidence = "score":    its per-frame dead-ball evidence, clipped
alpha = 0 is no fusion. alpha and the evidence type are chosen per fold on the training matches.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from ..model import deadball as DB
from . import boundaries, dataset, decode, degrade, labels, parse, splits, to_atd

EMPTY_TAGS = pd.DataFrame({"code_key": pd.Series(dtype=str), "type": pd.Series(dtype=str),
                           "start_frame": pd.Series(dtype=np.int64), "end_frame": pd.Series(dtype=np.int64)})


def run_detector(df, ev, speed, tags=EMPTY_TAGS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score, _, _ = DB.evidence(df, ev, tags, speed)
    dead = DB.decode(score, df["graded"].to_numpy(), df["known_dead"].to_numpy())
    return score.astype(np.float32), dead


def detector_idsse(mid, level="realistic", seed=1):
    """Detector output on a degraded IDSSE match (cached). The match is degraded again with the same
    seed, which reproduces the cached training table exactly (checked)."""
    path = os.path.join(parse.CACHE_DIR, "deadball_{}_{}_s{}.npz".format(mid, level, seed))
    if os.path.exists(path):
        z = np.load(path)
        return z["score"], z["dead"]
    m = parse.load_match(mid, quiet=True)
    lv = dataset.LEVELS[level]
    d = degrade.degrade(to_atd.clean_arrays(m), dataset.hazards(), m["status"] == 0, seed=seed,
                        scale_loss=dataset._scaled((0.5, 3.0), lv["loss"]),
                        scale_regain=dataset._scaled((2.5, 0.2), lv["regain"]))
    df, ev, speed = to_atd.build_frames(d, extra=True)
    table = dataset.match_table(mid, level, seed)
    same = np.array_equal(df["ball_observed"].to_numpy(), table["ball_observed"].to_numpy()) and \
        np.allclose(df["all_mean_speed"].to_numpy(float), table["all_mean_speed"].to_numpy(float), equal_nan=True)
    if not same:
        raise RuntimeError("rebuilt {} differs from its cached table".format(mid))
    score, dead = run_detector(df, ev, speed)
    np.savez(path, score=score, dead=dead)
    return score, dead


def fuse(logp, score, dead, kind, alpha):
    if alpha == 0:
        return logp
    ev = np.where(dead, 1.0, -1.0) if kind == "decision" else np.clip(score, -1, 1)
    out = logp.copy()
    out[:, labels.D] = out[:, labels.D] + alpha * ev
    return out


def cv(alphas=(0.0, 0.25, 0.5, 1.0, 2.0)):
    from . import bench
    cases = boundaries._cases()
    det = {}
    for v in cases:
        det[v] = detector_idsse(v)
        gt = cases[v][2]
        dd = det[v][1]
        print("detector on {}: dead frames right {:.0%}, live frames right {:.0%}".format(
            v, (dd[gt == labels.D]).mean(), (~dd[gt != labels.D]).mean()), flush=True)
    grid = decode.grid_scores({v: c[:3] for v, c in cases.items()})
    rows = []
    for v, (lp, h, gt, _) in cases.items():
        costs = decode.pick_costs(grid, [m for m in cases if m != v])
        for kind in ("decision", "score"):
            for a in alphas:
                if a == 0 and kind == "score":
                    continue
                codes = decode.viterbi(fuse(lp, *det[v], kind, a), h, **costs)
                s = bench.score(gt, codes, h)
                rows.append({"match": v, "setting": (kind, a), "spellF1": s["spellF1"], "macroF1": s["macroF1"]})
    scores = pd.DataFrame(rows)
    res = boundaries.cv_pick(scores)
    base = scores[scores.setting == ("decision", 0.0)].set_index("match")
    res["no_fusion"] = res.validate.map(base.spellF1)
    res["macro_fused"] = [scores[(scores.match == r.validate) & (scores.setting == r.setting)].macroF1.iloc[0]
                          for r in res.itertuples()]
    res["macro_no_fusion"] = res.validate.map(base.macroF1)
    print(scores.groupby("setting")[["spellF1", "macroF1"]].mean().round(3).to_string())
    print(res.round(3).to_string(index=False))
    print("cross-validated spellF1: no fusion {:.3f} -> fused {:.3f} | macroF1 {:.3f} -> {:.3f}".format(
        res.no_fusion.mean(), res.spellF1.mean(), res.macro_no_fusion.mean(), res.macro_fused.mean()))


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        cv()
