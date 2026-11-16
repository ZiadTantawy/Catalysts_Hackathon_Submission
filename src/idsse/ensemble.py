"""Combine tree and network probabilities, scored on the IDSSE folds.

    python -m src.idsse.ensemble lgbm_local nn_tcn nn_gru

Reads cache/idsse/cvpred_<name>.npz (validation-match probabilities from the cross-validation), averages
them with several weights, decodes with the bench smoother (costs picked per fold on its training
matches) and prints spellF1 / macroF1 per mix. An equal-weight average is the default choice: picking
the weight on these same five folds would flatter it, so a tuned weight is only preferred if it wins
clearly.
"""

import os
import sys
from itertools import product

import numpy as np
import pandas as pd

from . import decode, inputs, parse, splits


def load(name):
    z = np.load(os.path.join(parse.CACHE_DIR, "cvpred_{}.npz".format(name)))
    out = {}
    for v in splits.DEV:
        key = v if v in z.files else "{}_realistic".format(v)
        p = z[key]
        out[v] = np.exp(p) if p.max() <= 0 else p                # stored as log-probabilities or probabilities
    return out


def score_mix(preds, weights):
    cases = {}
    for v in splits.DEV:
        p = sum(w * preds[k][v] for k, w in weights.items())
        a = inputs.get(v, "realistic", 1)
        cases[v] = (np.log(np.clip(p, 1e-6, 1)), a["half"], a["y"])
    cv = decode.cv_scores(decode.grid_scores(cases), splits.cv_folds())
    return cv


def main(names):
    preds = {n: load(n) for n in names}
    rows = []
    grid = [w for w in product(*[np.linspace(0, 1, 5)] * len(names)) if abs(sum(w) - 1) < 1e-9]
    for ws in grid:
        weights = {n: w for n, w in zip(names, ws) if w > 0}
        cv = score_mix(preds, weights)
        rows.append({**{n: w for n, w in zip(names, ws)}, "spellF1": cv.spellF1.mean(), "macroF1": cv.macroF1.mean(),
                     **{"spell_" + v: s for v, s in zip(cv.validate, cv.spellF1)}})
        print({n: round(w, 2) for n, w in zip(names, ws)}, "spellF1 %.3f" % cv.spellF1.mean(), flush=True)
    res = pd.DataFrame(rows).sort_values("spellF1", ascending=False)
    res.to_csv(os.path.join(parse.CACHE_DIR, "ensemble_cv.csv"), index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).head(10).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
