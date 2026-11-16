"""The IDSSE bench: labels, the organisers' baseline on (degraded) IDSSE, and the official scores.

    gt = bench.variant_b_codes(mid, m)              # event-confirmed possession labels (notebook 02)
    pred = bench.baseline_labels(atd, window=50)    # starter/baseline.py, same algorithm
    bench.score(gt, pred, m["half"])                # spellF1 / macroF1 from evaluation/possession_metrics.py

The baseline and the scorer are imported from the organisers' files, not re-implemented, so a number
here means the same as on the leaderboard. Calibration target (EVALUATION.md, full BAR-ATH match):
baseline_smooth50 spellF1 0.2433, and spellF1 ranks smooth50 > smooth125 > smooth25 > raw.
"""

import os
import sys

import numpy as np

from . import events, labels, parse

ROOT = parse.REPO_ROOT
for p in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)
import baseline as _baseline  # noqa: E402  starter/baseline.py
import possession_metrics as _pm  # noqa: E402  evaluation/possession_metrics.py

HOME_VALUE = 1                     # BallPossession value of the home team (notebook 01)
WINDOWS = {"raw": 0, "smooth25": 25, "smooth50": 50, "smooth125": 125}


def variant_b_codes(mid, m):
    """H / A / D per frame (0 / 1 / 2): possession changes only where the event log confirms them."""
    f = parse.files()[mid]
    ev = events.read_events(f["events"], parse.read_matchinfo(f["info"]))
    off, _, _ = events.estimate_offset(m, ev)
    ctl = events.control_events(ev, m, off)
    raw = labels.frame_codes(m, HOME_VALUE)
    return events.event_confirmed_relabel(raw, m["half"], events.event_turnovers(ctl, raw))


SNAP_WINDOW = int(1.5 * 25)        # frames either side of a variant-B turnover searched for the touch
SNAP_RADIUS = 1.0                  # m: a player of the new team this close to the ball = a touch


def snapped_codes(mid, m):
    """Variant B with every live turnover moved to the new team's first touch in the CLEAN tracking
    (labels v3). Why: the BAR-ATH ground truth flips "near the new team's first contact" (docs/PLAN.md);
    variant B puts the change where DFL's flag changed, which on the IDSSE folds caps even a model on
    perfect tracking at spellF1 ~0.77. Dead ball and restarts are unchanged."""
    from . import boundaries
    codes = variant_b_codes(mid, m)
    near = labels.nearest_player_team(m, SNAP_RADIUS)
    sign = np.where(near == 0, 1, np.where(near == 1, -1, 0)).astype(np.int8)
    return boundaries.snap_to_contact(codes, sign, m["half"], SNAP_WINDOW)


def nearest_codes(atd):
    """Team (0 / 1) of the nearest tracked slot to the ball, -1 where the ball is not seen. Guessed
    positions count, as in starter/baseline.py (it reads every slot the ATD reports)."""
    p, b = atd["players"], atd["ball"]
    d = np.hypot(p[..., 0] - b[:, None, 0], p[..., 1] - b[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    j = d.argmin(axis=1)
    ok = np.isfinite(d[np.arange(len(j)), j])
    return np.where(ok, atd["slot_team"][j], _baseline.NOBODY).astype(np.int8)


def baseline_labels(atd, window):
    """Letters H / A / D, smoothed exactly as starter/baseline.py does (its own function)."""
    return _baseline.smooth(nearest_codes(atd), window)


def score(gt_codes, pred, half):
    """spellF1 (ranking metric) and macroF1 of a prediction against IDSSE labels.

    The halves are concatenated in IDSSE, so the first frame of each later half is set to X in the
    ground truth: the scorer then treats the halves as separate runs, as it does BAR-ATH's."""
    gt = gt_codes.astype(np.int8).copy()
    gt[np.flatnonzero(np.r_[False, half[1:] != half[:-1]])] = _pm.X
    pr = _pm.label_codes(pred) if pred.dtype.kind in "US" else pred.astype(np.int8)
    cm3 = _pm.scored_confusion(_pm.confusion(gt, pr))
    runs = _pm.valid_runs(gt)
    filled = _pm.fill_abstentions(pr, runs)
    spells = _pm.spell_scores(gt, filled, runs)
    return {"spellF1": spells["mean_f1"], "macroF1": _pm.macro_f1(cm3),
            "n_gt_spells": spells["n_gt"], "n_pred_spells": spells["n_pred"]}
