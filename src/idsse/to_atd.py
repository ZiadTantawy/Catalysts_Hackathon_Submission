"""IDSSE match -> ATD-shaped arrays -> the etl-v1.3 feature tables, through the ETL's own code.

    atd = to_atd.clean_arrays(m)          # perfect tracking in the ATD layout (degrader works on this)
    df, ev = to_atd.build_frames(atd)     # frames table with the same columns as cache/frames.parquet

Why: a model trained on IDSSE only transfers to BAR-ATH if both go through identical feature code.
So this module never computes a feature itself. It reproduces stages 3-7 of src/etl/build.py, calling
the same functions, with three IDSSE-specific inputs where build.py reads BAR-ATH files:

- timeline: every IDSSE frame is "graded" (the tracked halves are play, pre-match is not included);
- camera: a virtual camera (`virtual_camera`), since IDSSE has no video or homography;
- tagging: none, so the tagging columns exist with their neutral values (models must not use them).

`situations.compute` marks BAR-ATH's published pre-kickoff stretch of the 2nd half as known dead
(config.HALF_BLOCKS / KICKOFF_2ND); that is a fact about BAR-ATH only, so it is cleared here.

ATD layout (what src/etl/io_atd.py + orient.to_metres produce before orientation):
    players [T, 46, 3]  metres on a PITCH_L x PITCH_W pitch, origin at a corner; channel 2 = speed,
                        NaN speed = position guessed by the tracker (not measured)
    ball    [T, 3]      x, y in metres, channel 2 unknown in the ATD (NaN here)
    slot_team [46]      0 home / 1 away;  slot_is_gk [46]
"""

import warnings

import numpy as np
import pandas as pd

from ..etl import ball, features, io_tagging, orient, players, situations
from ..etl import config as C

N_SLOTS = 46
DFL_L, DFL_W = 105.0, 68.0


def to_pitch(x, y):
    """DFL metres (origin at the centre spot) -> ETL metres (origin at a corner, PITCH_L x PITCH_W)."""
    return (x + DFL_L / 2) * (C.PITCH_L / DFL_L), (y + DFL_W / 2) * (C.PITCH_W / DFL_W)


def clean_arrays(m):
    """Perfect IDSSE tracking in the ATD layout: one slot per person, every position measured."""
    T, P = m["xy"].shape[:2]
    if P > N_SLOTS:
        raise ValueError("{} persons do not fit {} slots".format(P, N_SLOTS))
    p = np.full((T, N_SLOTS, 3), np.nan, np.float32)
    p[:, :P, 0], p[:, :P, 1] = to_pitch(m["xy"][..., 0], m["xy"][..., 1])
    p[:, :P, 2] = np.where(np.isfinite(m["xy"][..., 0]), m["speed"], np.nan)
    b = np.full((T, 3), np.nan, np.float32)
    b[:, 0], b[:, 1] = to_pitch(m["ball"][:, 0], m["ball"][:, 1])
    slot_team = np.zeros(N_SLOTS, np.int64)
    slot_team[:P] = m["person_team"]
    slot_is_gk = np.zeros(N_SLOTS, bool)
    slot_is_gk[:P] = m["person_gk"]
    atd = {"players": p, "ball": b, "slot_team": slot_team, "slot_is_gk": slot_is_gk,
           "half": m["half"].astype(np.int8)}
    atd["cam"] = virtual_camera(atd)
    return atd


def virtual_camera(atd, width_m=62.4, smooth_s=1.0, follow_ball=0.5, follow_y=0.2):
    """A tactical camera from the tracking itself, in the same units as io_homography.camera_features
    (centre normalised 0-1, unoriented). Centre x follows a mix of the ball and the players' mean x,
    centre y stays near the middle; both smoothed. Defaults from BAR-ATH's cache/report.txt: view
    width p50 62.4 m; camera x correlates 0.95 with the team blocks and 0.83 with the ball; camera y
    hardly follows the ball (corr 0.18). A position is on screen when inside the view; the view's
    height is the pitch-width share the tactical camera covers at that zoom (~0.65 of its width)."""
    p, b = atd["players"], atd["ball"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # frames with nobody tracked
        px = np.nanmean(p[..., 0], axis=1)
    px = np.where(np.isfinite(px), px, C.PITCH_L / 2)
    bx = np.where(np.isfinite(b[:, 0]), b[:, 0], px)
    by = np.where(np.isfinite(b[:, 1]), b[:, 1], C.PITCH_W / 2)
    cx = follow_ball * bx + (1 - follow_ball) * px
    cy = follow_y * by + (1 - follow_y) * C.PITCH_W / 2
    k = max(int(smooth_s * C.FPS), 1)
    cx = pd.Series(cx).groupby(atd["half"]).transform(lambda s: s.ewm(span=k).mean()).to_numpy()
    cy = pd.Series(cy).groupby(atd["half"]).transform(lambda s: s.ewm(span=k).mean()).to_numpy()
    cx = np.clip(cx, width_m / 2, C.PITCH_L - width_m / 2)

    half_w, half_h = width_m / 2, 0.65 * width_m / 2
    tracked = np.isfinite(p[..., 0])
    on = tracked & (np.abs(p[..., 0] - cx[:, None]) <= half_w) & (np.abs(p[..., 1] - cy[:, None]) <= half_h)
    n_tr = tracked.sum(1)
    b_known = np.isfinite(b[:, 0])
    b_on = np.full(len(b), -1, np.int8)
    b_on[b_known] = ((np.abs(b[b_known, 0] - cx[b_known]) <= half_w)
                     & (np.abs(b[b_known, 1] - cy[b_known]) <= half_h)).astype(np.int8)
    pan = np.full(len(cx), np.nan)
    j = 5
    pan[j:-j] = np.hypot(cx[2 * j:] - cx[:-2 * j], cy[2 * j:] - cy[:-2 * j]) / (2 * j / C.FPS)
    return {
        "cam_cx_n": (cx / C.PITCH_L).astype(np.float32),
        "cam_cy_n": (cy / C.PITCH_W).astype(np.float32),
        "cam_view_width_m": np.full(len(cx), width_m, np.float32),
        "cam_pan_speed": pan.astype(np.float32),
        "cam_reproj_error": np.zeros(len(cx), np.float32),
        "n_on_screen": on.sum(1).astype(np.int8),
        "frac_tracked_off_screen": np.where(n_tr > 0, 1 - on.sum(1) / np.maximum(n_tr, 1),
                                            np.nan).astype(np.float32),
        "ball_on_screen": b_on,
        "_on_screen": on,          # per position, for the degrader; dropped before the frames table
    }


def timeline(half):
    n = len(half)
    phase = np.where(half == 1, orient.PHASE_H1, orient.PHASE_H2).astype(np.int8)
    t_half = np.zeros(n, np.float32)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        t_half[idx] = np.arange(len(idx)) / C.FPS
    return {"half": half.astype(np.int8), "phase": phase, "graded": half > 0, "t_half_s": t_half}


_EMPTY_TAGS = pd.DataFrame({"code_key": pd.Series(dtype=str), "side": pd.Series(dtype=str),
                            "start_frame": pd.Series(dtype=np.int64), "end_frame": pd.Series(dtype=np.int64),
                            "type": pd.Series(dtype=str)})


def build_frames(atd):
    """Stages 3-7 of src/etl/build.py on ATD-shaped arrays. Returns (frames table, events table)."""
    half = atd["half"]
    n = len(half)
    tl = timeline(half)
    cam = {k: v for k, v in atd["cam"].items() if not k.startswith("_")}

    flips = orient.detect_flips(atd["players"], atd["slot_team"], half)
    p_o, b_o, flip_frame = orient.apply_orientation(atd["players"], atd["ball"], tl["phase"], flips)
    cx_m = cam.pop("cam_cx_n") * C.PITCH_L
    cy_m = cam.pop("cam_cy_n") * C.PITCH_W
    cam["cam_cx"] = np.where(flip_frame, C.PITCH_L - cx_m, cx_m).astype(np.float32)
    cam["cam_cy"] = np.where(flip_frame, C.PITCH_W - cy_m, cy_m).astype(np.float32)

    ball_cols, _ = ball.clean(b_o, half)
    cam["cam_d_ball"] = np.hypot(cam["cam_cx"] - ball_cols["ball_x"],
                                 cam["cam_cy"] - ball_cols["ball_y"]).astype(np.float32)

    pl, _ = players.prepare(p_o, atd["slot_team"], atd["slot_is_gk"])
    feat, near, _, _ = features.team_contest_deadball(p_o, pl, ball_cols)
    con_cols, con_episodes, _ = features.contacts(feat, half, ball_cols)
    feat.update(con_cols)
    sit_cols, _ = situations.compute(pl, feat, ball_cols, tl)
    sit_cols["known_dead"] = np.zeros(n, bool)          # BAR-ATH-only published fact, see docstring
    tag_cols, _ = io_tagging.per_frame(_EMPTY_TAGS, n)

    df = pd.DataFrame({"frame": np.arange(n, dtype=np.int32), **tl, "rotated": flip_frame,
                       **ball_cols, **feat, **sit_cols, **cam, **tag_cols})
    df = pd.concat([df, features.context_windows(df)], axis=1)
    ev, _ = ball.events(ball_cols, half, near, con_episodes)
    _, run_id = ball.runs_table(ball_cols, half)
    df["ball_run_id"] = run_id
    return df, ev
