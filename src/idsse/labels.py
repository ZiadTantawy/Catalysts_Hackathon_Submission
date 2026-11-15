"""Frame labels from IDSSE ball flags, possession spells, and the controlled-possession relabel.

Codes follow the submission letters: 0 = H, 1 = A, 2 = D (LETTERS[code] gives the letter).

    codes = labels.frame_codes(m, home_value)       # from BallStatus / BallPossession
    sp = labels.spells(codes, m["half"])            # one row per run, per half
    codes2 = labels.give_back(codes, m["half"], tau_frames=50)
"""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(list("HAD"))


def nearest_player_team(m, max_dist=1.0):
    """Team (0 home / 1 away) of the player nearest the ball where they are within max_dist m, else -1."""
    d = np.hypot(m["xy"][..., 0] - m["ball"][:, None, 0], m["xy"][..., 1] - m["ball"][:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    j = d.argmin(axis=1)
    near = d[np.arange(len(j)), j] <= max_dist
    return np.where(near, m["person_team"][j], -1)


def frame_codes(m, home_value):
    """H / A from BallPossession while BallStatus is alive, D where dead."""
    codes = np.where(m["poss"] == home_value, H, A).astype(np.int8)
    codes[m["status"] == 0] = D
    return codes


def runs(v):
    """(start, end_exclusive, value) of each maximal run in v."""
    v = np.asarray(v)
    if len(v) == 0:
        return np.zeros((0, 3), np.int64)
    cut = np.flatnonzero(v[1:] != v[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(v)]
    return np.stack([starts, ends, v[starts]], axis=1)


def spells(codes, half):
    """One row per run of the same code inside a half: start, end, code, half, frames."""
    rows = []
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        r = runs(codes[idx])
        r[:, :2] += idx[0]
        rows.append(np.c_[r, np.full(len(r), h), r[:, 1] - r[:, 0]])
    return np.concatenate(rows)


def give_back(codes, half, tau_frames, before_dead=False):
    """Controlled-possession relabel: a spell of one team shorter than tau_frames, with the other team
    directly on both sides (no dead ball between), goes back to that team. Repeated until stable,
    because merging can expose a new short spell.

    before_dead: also give back a short spell that sits between the other team and a dead ball (a
    touch or deflection just before the ball goes out). A short spell right after a dead ball is never
    changed: after a restart the restarting team has the ball (CHALLENGE.md, "Starting point")."""
    out = codes.copy()
    while True:
        changed = False
        for h in np.unique(half):
            idx = np.flatnonzero(half == h)
            r = runs(out[idx])
            skip = False
            for i in range(1, len(r)):
                if skip:  # its left neighbour just changed, so r is stale here; next pass sees it
                    skip = False
                    continue
                s, e, c = r[i]
                prev_c = r[i - 1, 2]
                next_c = r[i + 1, 2] if i + 1 < len(r) else D
                if c == D or prev_c == D or prev_c == c or e - s >= tau_frames:
                    continue
                if next_c == prev_c or (before_dead and next_c == D):
                    out[idx[s:e]] = prev_c
                    changed = skip = True
        if not changed:
            return out
