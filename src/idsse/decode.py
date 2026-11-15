"""The bench smoother: per-frame probabilities -> clean possessions (Viterbi over H / A / D).

    codes = decode.viterbi(logp, half, team_cost=12.0, dead_cost=8.0)

Per half, it picks the sequence of states that maximises the summed per-frame log-probabilities minus
a cost for every change: `team_cost` for H <-> A, `dead_cost` for going dead or coming back. A change
therefore needs consistent evidence over a stretch of frames, so a few wrong frames cannot create a
possession. The costs are chosen per cross-validation fold on its training matches only
(grid_scores -> cv_scores), never on the validation or test match being scored.

This is the IDSSE bench's decoder, so models can be scored with spellF1. The other half's labeller has
its own (semi-Markov) decoder; what we hand over is the per-frame evidence, not these labels.
"""

import itertools

import numpy as np
import pandas as pd

from . import bench

H, A, D = 0, 1, 2


def cost_matrix(team_cost, dead_cost):
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = team_cost
    c[H, D] = c[D, H] = c[A, D] = c[D, A] = dead_cost
    return c


def _viterbi_segment(lp, c):
    n = len(lp)
    back = np.empty((n, 3), np.int8)
    d0, d1, d2 = lp[0]
    c = c.tolist()
    lp = lp.tolist()
    for t in range(1, n):
        new = []
        for j in range(3):
            a0, a1, a2 = d0 - c[0][j], d1 - c[1][j], d2 - c[2][j]
            if a0 >= a1 and a0 >= a2:
                best, arg = a0, 0
            elif a1 >= a2:
                best, arg = a1, 1
            else:
                best, arg = a2, 2
            back[t, j] = arg
            new.append(best + lp[t][j])
        d0, d1, d2 = new
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax([d0, d1, d2]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi(logp, half, team_cost=12.0, dead_cost=8.0):
    """logp [T, 3]: log P(H), log P(A), log P(D) per frame. Decoded separately in each half."""
    c = cost_matrix(team_cost, dead_cost)
    out = np.empty(len(logp), np.int8)
    starts = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1, len(half)]
    for a, b in zip(starts[:-1], starts[1:]):
        out[a:b] = _viterbi_segment(np.asarray(logp[a:b], np.float64), c)
    return out


TEAM_GRID = (16, 32, 64, 128, 256)
DEAD_GRID = (16, 32, 64, 128, 256, 512)


def grid_scores(cases, team_grid=TEAM_GRID, dead_grid=DEAD_GRID):
    """cases: {match: (logp, half, gt_codes)}. Scores every match at every cost pair, once, so that
    cross-validation can pick costs on the training matches of each fold without decoding again."""
    rows = []
    for mid, (lp, h, gt) in cases.items():
        for tc, dc in itertools.product(team_grid, dead_grid):
            s = bench.score(gt, viterbi(lp, h, tc, dc), h)
            rows.append({"match": mid, "team_cost": tc, "dead_cost": dc, **s})
    return pd.DataFrame(rows)


def pick_costs(grid, matches):
    """Cost pair with the best mean spellF1 over `matches` (the training side of a fold)."""
    g = grid[grid.match.isin(matches)].groupby(["team_cost", "dead_cost"]).spellF1.mean()
    tc, dc = g.idxmax()
    return {"team_cost": tc, "dead_cost": dc}


def cv_scores(grid, folds):
    """Per fold: costs picked on its training matches, score on its validation match."""
    out = []
    for train, val in folds:
        costs = pick_costs(grid, train)
        row = grid[(grid.match == val) & (grid.team_cost == costs["team_cost"])
                   & (grid.dead_cost == costs["dead_cost"])].iloc[0]
        out.append({"validate": val, **costs, "spellF1": row.spellF1, "macroF1": row.macroF1,
                    "n_pred_spells": row.n_pred_spells, "n_gt_spells": row.n_gt_spells})
    return pd.DataFrame(out)
