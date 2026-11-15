"""A virtual tactical camera for IDSSE, which has no video: where it points and what it sees.

Fitted on BAR-ATH's homography features (cache/frames.parquet, graded frames, no labels):

    cam_cx - L/2 = 0.91 * (team blocks' mid-point - L/2) + 0.10 * (ball - L/2)    residual sd 4.8 m
    view width p10 / p50 / p90 = 54.6 / 62.4 / 72.8 m; centre y p50 36.0 m
    pan speed p50 / p75 / p90 / p95 = 0.11 / 1.36 / 3.08 / 4.17 m/s: still half the time, then pans

So the camera is a lazy operator: it aims at that target and only pans when the target leaves a
dead band around the image centre, at a speed proportional to the error (capped).

    cx, cy, width = camera.track(players_m, ball_m, slot_team, half)
    cam = camera.features(players_m, ball_m, cx, cy, width)    # same keys as io_homography
"""

import warnings

import numpy as np

from ..etl import config as C

FPS = C.FPS
W_BLOCKS, W_BALL = 0.91, 0.10
WIDTH_M = 62.4
CY_M = 36.0


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def target_x(players, ball, slot_team):
    """Where the operator aims, along the pitch: mostly the team blocks' mid-point, a little the ball."""
    hx = _nanmean(players[:, slot_team == 0, 0], 1)
    ax = _nanmean(players[:, slot_team == 1, 0], 1)
    mid = np.where(np.isfinite(hx) & np.isfinite(ax), (hx + ax) / 2, np.where(np.isfinite(hx), hx, ax))
    mid = np.where(np.isfinite(mid), mid, C.PITCH_L / 2)
    bx = np.where(np.isfinite(ball[:, 0]), ball[:, 0], mid)
    return C.PITCH_L / 2 + W_BLOCKS * (mid - C.PITCH_L / 2) + W_BALL * (bx - C.PITCH_L / 2)


def track(players, ball, slot_team, half, deadband_m=3.0, gain=1.6, max_pan=6.0, width_m=WIDTH_M):
    """Camera centre per frame (unoriented metres) and view width. Restarts at each half."""
    tgt = target_x(players, ball, slot_team)
    cx = np.empty(len(tgt), np.float64)
    dt = 1.0 / FPS
    for i in range(len(tgt)):
        if i == 0 or half[i] != half[i - 1]:
            cx[i] = tgt[i]
            continue
        err = tgt[i] - cx[i - 1]
        v = 0.0 if abs(err) <= deadband_m else np.clip(gain * (err - np.sign(err) * deadband_m), -max_pan, max_pan)
        cx[i] = cx[i - 1] + v * dt
    cx = np.clip(cx, width_m / 2 - 10, C.PITCH_L - width_m / 2 + 10)
    return cx, np.full(len(cx), CY_M), np.full(len(cx), width_m)


def in_view(x, y, cx, cy, width, height_factor):
    """Positions inside the image: a rectangle width x (height_factor * width) around the centre."""
    return (np.abs(x - cx) <= width / 2) & (np.abs(y - cy) <= height_factor * width / 2)


def features(players, ball, cx, cy, width, height_factor=1.0):
    """Camera columns in the units of io_homography.camera_features (centre normalised, unoriented)."""
    cx_b, cy_b, w_b = cx[:, None], cy[:, None], width[:, None]
    tracked = np.isfinite(players[..., 0])
    on = tracked & in_view(players[..., 0], players[..., 1], cx_b, cy_b, w_b, height_factor)
    n_tr = tracked.sum(1)
    b_known = np.isfinite(ball[:, 0])
    b_on = np.full(len(ball), -1, np.int8)
    b_on[b_known] = in_view(ball[b_known, 0], ball[b_known, 1], cx[b_known], cy[b_known], width[b_known],
                            height_factor).astype(np.int8)
    pan = np.full(len(cx), np.nan)
    j = 5
    pan[j:-j] = np.hypot(cx[2 * j:] - cx[:-2 * j], cy[2 * j:] - cy[:-2 * j]) / (2 * j / FPS)
    return {
        "cam_cx_n": (cx / C.PITCH_L).astype(np.float32),
        "cam_cy_n": (cy / C.PITCH_W).astype(np.float32),
        "cam_view_width_m": width.astype(np.float32),
        "cam_pan_speed": pan.astype(np.float32),
        "cam_reproj_error": np.zeros(len(cx), np.float32),
        "n_on_screen": on.sum(1).astype(np.int8),
        "frac_tracked_off_screen": np.where(n_tr > 0, 1 - on.sum(1) / np.maximum(n_tr, 1),
                                            np.nan).astype(np.float32),
        "ball_on_screen": b_on,
    }
