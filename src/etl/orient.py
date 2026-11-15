"""Stages 2-3: the match timeline, metres, and one attacking direction for the whole match.

After this stage every coordinate is in metres on a 105.3 x 68 pitch, rotated so that the HOME team
attacks towards +x in both halves. Without that, every positional feature means the opposite thing
after halftime.

The direction is read from the data, not from the XML: in each half, a team's deepest outfield
shape and its goalkeeper sit near the goal it defends. For each half we compare the two hypotheses
("home defends x=0" vs "home defends x=L") using the median, over the whole half, of how close each
team's extreme player is to each goal line. The margin between the hypotheses goes into the report,
so a weak call is visible instead of silent.
"""

import warnings

import numpy as np

from . import config as C

PHASE_PRE, PHASE_H1, PHASE_HT, PHASE_H2, PHASE_POST = 0, 1, 2, 3, 4


def timeline(n=C.N_FRAMES):
    """half (0/1/2), phase (pre/H1/HT/H2/post), graded flag, seconds since half start."""
    half = np.zeros(n, dtype=np.int8)
    phase = np.zeros(n, dtype=np.int8)
    t_half = np.full(n, np.nan, dtype=np.float32)
    (h1, a1, b1), (h2, a2, b2) = C.HALF_BLOCKS
    half[a1:b1 + 1] = 1
    half[a2:b2 + 1] = 2
    phase[:a1] = PHASE_PRE
    phase[a1:b1 + 1] = PHASE_H1
    phase[b1 + 1:a2] = PHASE_HT
    phase[a2:b2 + 1] = PHASE_H2
    phase[b2 + 1:] = PHASE_POST
    t_half[a1:b1 + 1] = np.arange(b1 - a1 + 1) / C.FPS
    t_half[a2:b2 + 1] = np.arange(b2 - a2 + 1) / C.FPS
    return {"half": half, "phase": phase, "graded": half > 0, "t_half_s": t_half}


def to_metres(players, ball):
    """Scale x, y channels (0-1) to metres in place-safe copies. Channel 0 = x, 1 = y."""
    p = players.copy()
    b = ball.copy()
    p[..., 0] *= C.PITCH_L
    p[..., 1] *= C.PITCH_W
    b[:, 0] *= C.PITCH_L
    b[:, 1] *= C.PITCH_W
    return p, b


def detect_flips(p_m, slot_team, half):
    """{half: (flip, margin_m)}; flip=True means raw home attacks -x in that half."""
    out = {}
    x = p_m[..., 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for h in (1, 2):
            rows = half == h
            xh = x[rows][:, slot_team == 0]
            xa = x[rows][:, slot_team == 1]
            enough = (np.isfinite(xh).sum(1) >= 6) & (np.isfinite(xa).sum(1) >= 6)
            if enough.sum() < 100:
                out[h] = (False, 0.0)
                continue
            h_min, h_max = np.nanmin(xh[enough], 1), np.nanmax(xh[enough], 1)
            a_min, a_max = np.nanmin(xa[enough], 1), np.nanmax(xa[enough], 1)
            # hypothesis A: home defends x=0 (home min near 0, away max near L)
            cost_a = np.median(h_min) + np.median(C.PITCH_L - a_max)
            # hypothesis B: home defends x=L
            cost_b = np.median(C.PITCH_L - h_max) + np.median(a_min)
            out[h] = (bool(cost_b < cost_a), float(abs(cost_a - cost_b)))
    return out


def apply_orientation(p_m, b_m, phase, flips):
    """Rotate 180 deg wherever home attacks -x. Pre-match follows H1, HT and post follow nearest."""
    flip_frame = np.zeros(len(phase), dtype=bool)
    f1, f2 = flips[1][0], flips[2][0]
    flip_frame[(phase == PHASE_PRE) | (phase == PHASE_H1) | (phase == PHASE_HT)] = f1
    flip_frame[(phase == PHASE_H2) | (phase == PHASE_POST)] = f2
    p = p_m.copy()
    b = b_m.copy()
    p[flip_frame, :, 0] = C.PITCH_L - p[flip_frame, :, 0]
    p[flip_frame, :, 1] = C.PITCH_W - p[flip_frame, :, 1]
    b[flip_frame, 0] = C.PITCH_L - b[flip_frame, 0]
    b[flip_frame, 1] = C.PITCH_W - b[flip_frame, 1]
    return p, b, flip_frame
