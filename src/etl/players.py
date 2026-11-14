"""Stage 5a: which player positions to trust, and which team each one belongs to, frame by frame.

Decisions are in docs/PLAN.md; every rule here is generic and fires wherever its condition holds.

  measured   (D1)  position AND speed present. The tracker leaves the speed empty when it did not see
                   the player and is holding or predicting the position. Only measured players may be
                   nearest to the ball, in contact or in control (D2).
  team       (P6)  per-frame team, int8: 0 home · 1 away · -1 dropped. Starts from the slot's team;
                   a measured keeper-slot track standing at a goal belongs to the team defending it
                   (the tracker hands goalkeeper detections to either keeper slot, confirmed on video
                   in both directions). An unmeasured keeper-slot position in the opponent's half is a
                   ghost and is dropped.
  keeper     (D3)  per-frame keeper role, decided from position, never from the slot number.
  dup        (P8)  same-team positions within DUP_RADIUS: one person tracked twice. The measured one is
                   kept over a guess; between equals, the lower slot.
  xdup       (P9)  opposite-team positions that are one object (tighter than any marking), sustained.
                   Never deleted; ball attribution through them is made neutral in features.py.
  far / off        people far outside the pitch are dropped; just outside are kept for ball distance.

Coordinates are oriented metres (home defends x = 0, attacks +x).
"""

import numpy as np

from . import config as C
from .util import out_dist, runs_true


def _in_box(x, y, goal_x):
    depth = np.abs(x - goal_x)
    return (depth <= C.BOX_DEPTH) & (np.abs(y - C.PITCH_W / 2) <= C.BOX_HALF_WIDTH)


def keeper_team(X, Y, meas, team, slot_team, slot_is_gk):
    """P6 in place on `team`. Returns (n_frames_changed, n_ghosts_dropped) per keeper slot."""
    usable = meas & (team >= 0)
    changed, ghosts = {}, {}
    for k in np.flatnonzero(slot_is_gk):
        own = int(slot_team[k])
        before = team[:, k].copy()
        for goal_x, defender in ((0.0, C.TEAM_IDX_HOME), (C.PITCH_L, C.TEAM_IDX_AWAY)):
            depth = np.abs(X[:, k] - goal_x)
            near_goal = meas[:, k] & (depth <= C.KEEPER_GOAL_ZONE)
            others = usable.copy()
            others[:, k] = False
            od = np.where(others, np.abs(X - goal_x), np.inf)
            deepest = depth < od.min(axis=1)
            crowd = (others & _in_box(X, Y, goal_x)).sum(axis=1)
            fire = near_goal & deepest & (crowd < C.KEEPER_BOX_CROWD)
            team[fire, k] = defender
        changed[int(k) + 1] = int((team[:, k] != before).sum())
        # an unmeasured keeper-slot position in the opponent's half: a ghost, not a person
        x_att = X[:, k] if own == C.TEAM_IDX_HOME else C.PITCH_L - X[:, k]
        ghost = np.isfinite(X[:, k]) & ~meas[:, k] & (x_att > C.PITCH_L / 2)
        team[ghost, k] = -1
        ghosts[int(k) + 1] = int(ghost.sum())
    return changed, ghosts


def keeper_role(X, Y, meas, team, slot_is_gk):
    """[T, S] bool: the one player per team and frame acting as goalkeeper, from position."""
    n, s = X.shape
    role = np.zeros((n, s), dtype=bool)
    rows = np.arange(n)
    for t, goal_x in ((C.TEAM_IDX_HOME, 0.0), (C.TEAM_IDX_AWAY, C.PITCH_L)):
        mine = meas & (team == t)
        # a keeper slot carrying this team's tag on this team's side of the pitch
        slot_ok = mine & slot_is_gk[None, :] & (np.abs(X - goal_x) < C.PITCH_L / 2)
        has_slot = slot_ok.any(axis=1)
        role[rows[has_slot], np.argmax(slot_ok, axis=1)[has_slot]] = True
        # otherwise the deepest measured player of the team inside its own penalty area
        cand = mine & _in_box(X, Y, goal_x)
        depth = np.where(cand, np.abs(X - goal_x), np.inf)
        j = np.argmin(depth, axis=1)
        use = ~has_slot & np.isfinite(depth[rows, j])
        role[rows[use], j[use]] = True
    return role


def _velocity(X, Y):
    k = C.PLAYER_VEL_K
    vx = np.full(X.shape, np.nan, dtype=np.float32)
    vy = np.full(X.shape, np.nan, dtype=np.float32)
    vx[k:-k] = (X[2 * k:] - X[:-2 * k]) / (2 * k / C.FPS)
    vy[k:-k] = (Y[2 * k:] - Y[:-2 * k]) / (2 * k / C.FPS)
    return vx, vy


def duplicates(X, Y, meas, team):
    """(dup [T,S] same-team second copies, xdup [T,S] opposite-team one-object pairs, raw xdup)."""
    n, s = X.shape
    vx, vy = _velocity(X, Y)
    dup = np.zeros((n, s), dtype=bool)
    xraw = np.zeros((n, s), dtype=bool)
    lower = np.tril(np.ones((s, s), dtype=bool), k=-1)[None]          # j < i
    for a0 in range(0, n, C.PAIR_CHUNK):
        a1 = min(n, a0 + C.PAIR_CHUNK)
        x, y, m, tm = X[a0:a1], Y[a0:a1], meas[a0:a1], team[a0:a1]
        ok = np.isfinite(x) & (tm >= 0)
        d = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        pair = ok[:, :, None] & ok[:, None, :]
        same = (tm[:, :, None] == tm[:, None, :]) & pair
        close = same & (d < C.DUP_RADIUS)
        # i is the copy to drop if a close partner j is measured while i is not, or both share the
        # same status and j has the lower slot index
        mi, mj = m[:, :, None], m[:, None, :]
        better = (mj & ~mi) | ((mj == mi) & lower)
        dup[a0:a1] = (close & better).any(axis=2)
        dv = np.hypot(vx[a0:a1, :, None] - vx[a0:a1, None, :],
                      vy[a0:a1, :, None] - vy[a0:a1, None, :])
        opp = pair & ~same & mi & mj & (d < C.XDUP_DIST) & (dv < C.XDUP_RELV)
        xraw[a0:a1] = opp.any(axis=2)
    xdup = np.zeros_like(xraw)
    for k in range(s):
        st, en = runs_true(xraw[:, k])
        for a, b in zip(st, en):
            if b - a >= C.XDUP_MIN:
                xdup[a:b, k] = True
    return dup, xdup


def prepare(p, slot_team, slot_is_gk):
    """p: [T, S, C] oriented metres (channel 2 = the tracker's speed). Returns a dict of [T, S] arrays
    and aggregate stats."""
    X = p[..., 0].astype(np.float32)
    Y = p[..., 1].astype(np.float32)
    present = np.isfinite(X) & np.isfinite(Y)
    meas = present & np.isfinite(p[..., 2]) if p.shape[2] >= 3 else present.copy()
    team = np.where(present, slot_team[None, :], -1).astype(np.int8)

    with np.errstate(invalid="ignore"):
        o = out_dist(X, Y)
    far = present & (o > C.FAR_OFF_M)
    off = present & (o > C.OFF_PITCH_M) & ~far
    team[far] = -1

    changed, ghosts = keeper_team(X, Y, meas, team, slot_team, slot_is_gk)
    dup, xdup = duplicates(X, Y, meas, team)
    team[dup] = -1
    keeper = keeper_role(X, Y, meas, team, slot_is_gk)

    stats = {
        "positions_present": int(present.sum()),
        "positions_measured": int(meas.sum()),
        "far_off_points_dropped": int(far.sum()),
        "off_pitch_points": int(off.sum()),
        "duplicate_track_points": int(dup.sum()),
        "keeper_team_changed": changed,
        "keeper_ghosts_dropped": ghosts,
        "xdup_points": int(xdup.sum()),
    }
    return {"X": X, "Y": Y, "present": present, "meas": meas, "team": team, "keeper": keeper,
            "off": off, "dup": dup, "xdup": xdup}, stats
