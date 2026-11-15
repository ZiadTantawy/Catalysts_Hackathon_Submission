"""Model-ready arrays per table, cached: inputs, their home/away mirror, labels, halves.

    a = inputs.get("J03WOH", "realistic", 1)     # dict: X, Xm, y, ym, half, feats
    a["X"][::10]                                 # every 10th frame, for training

X uses dataset.feature_columns of the table with prep.add_long_context; Xm is the same frames seen with
home and away exchanged (prep.swap_home_away), ym the exchanged labels. Every model and every fold
reads these, so features are computed once and are identical everywhere.
"""

import os

import numpy as np

from . import dataset, parse, prep

INPUT_DIR = os.path.join(parse.CACHE_DIR, "inputs")


def get(mid, level, seed, force=False):
    path = os.path.join(INPUT_DIR, "{}_{}_s{}.npz".format(mid, level, seed))
    if os.path.exists(path) and not force:
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files} | {"feats": list(z["feats"])}
    df = prep.add_long_context(dataset.match_table(mid, level, seed))
    feats = dataset.feature_columns(df)
    sw = prep.swap_home_away(df)
    out = {"X": df[feats].to_numpy(np.float32), "Xm": sw[feats].to_numpy(np.float32),
           "y": df["label"].to_numpy(np.int8), "ym": sw["label"].to_numpy(np.int8),
           "half": df["half"].to_numpy(np.int8), "feats": np.array(feats)}
    os.makedirs(INPUT_DIR, exist_ok=True)
    np.savez(path, **out)
    out["feats"] = feats
    return out


def train_arrays(matches, levels, seed, step):
    """Stacked training rows (every `step`-th frame, plus the mirror of each) from several tables."""
    X, y = [], []
    for mid in matches:
        for level in levels:
            a = get(mid, level, seed)
            X += [a["X"][::step], a["Xm"][::step]]
            y += [a["y"][::step], a["ym"][::step]]
    return np.concatenate(X), np.concatenate(y)
