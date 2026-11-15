"""Camera features from the tactical homography: where the camera is looking, and how it moves.

The homography maps normalised pitch (x, y) -> image pixels (1920 x 1080). Its inverse takes the
image centre back to the pitch: the point the camera operator is framing. The tactical camera pans
to follow play, so that point tracks the action even when the ball itself is not detected -- which
is 41% of graded frames, often in multi-minute stretches. And when play stops, the camera usually
stops: pan speed is a dead-ball cue that does not depend on the ball at all.

Columns produced (pitch coordinates are oriented later, in build.py, with the same rotation as the
tracking):
    cam_cx_n, cam_cy_n      image centre on the pitch, normalised 0-1 (unoriented)
    cam_view_width_m        pitch distance spanned by the image's horizontal mid-line (zoom)
    cam_reproj_error        the file's own fit quality, lower is better
    n_on_screen / frac_tracked_off_screen    tracked players projected inside the image
    ball_on_screen          1 / 0 / -1 unknown
"""

import numpy as np
import pandas as pd

from . import config as C


def load(path, n=C.N_FRAMES):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if len(df) != n or not np.array_equal(df["frame"].to_numpy(), np.arange(n)):
        raise ValueError("{}: expected frames 0..{}".format(path, n - 1))
    H = df[["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]].to_numpy(
        np.float64).reshape(n, 3, 3)
    valid = df["valid"].to_numpy() == 1
    valid &= np.isfinite(H).all(axis=(1, 2))
    err = df["reprojection_error"].to_numpy(np.float64)
    return H, valid, err


def _apply(M, x, y):
    """Apply per-frame 3x3 M [T,3,3] to points x, y of shape [T] or [T, K]."""
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
        squeeze = True
    else:
        squeeze = False
    u = M[:, 0, 0, None] * x + M[:, 0, 1, None] * y + M[:, 0, 2, None]
    v = M[:, 1, 0, None] * x + M[:, 1, 1, None] * y + M[:, 1, 2, None]
    w = M[:, 2, 0, None] * x + M[:, 2, 1, None] * y + M[:, 2, 2, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = u / w, v / w
    if squeeze:
        return u[:, 0], v[:, 0]
    return u, v


def camera_features(H, valid, err, players_n, ball_n):
    """players_n [T,S,>=2], ball_n [T,>=2]: RAW normalised coordinates (before orientation)."""
    n = len(H)
    Hinv = np.full_like(H, np.nan)
    if valid.any():
        Hinv[valid] = np.linalg.inv(H[valid])
    cw, ch = C.IMAGE_W / 2.0, C.IMAGE_H / 2.0
    ones = np.ones(n)
    cx, cy = _apply(Hinv, cw * ones, ch * ones)
    lx, ly = _apply(Hinv, 0.0 * ones, ch * ones)
    rx, ry = _apply(Hinv, C.IMAGE_W * ones, ch * ones)
    width = np.hypot((rx - lx) * C.PITCH_L, (ry - ly) * C.PITCH_W)

    plausible = np.isfinite(cx) & (cx > -0.5) & (cx < 1.5) & (cy > -0.5) & (cy < 1.5)
    cx = np.where(plausible, cx, np.nan)
    cy = np.where(plausible, cy, np.nan)
    width = np.where(plausible & (width < 300), width, np.nan)

    # pan speed in m/s over +-5 frames, on the pitch
    pan = np.full(n, np.nan)
    k = 5
    pan[k:-k] = np.hypot((cx[2 * k:] - cx[:-2 * k]) * C.PITCH_L,
                         (cy[2 * k:] - cy[:-2 * k]) * C.PITCH_W) / (2 * k / C.FPS)

    u, v = _apply(H, players_n[..., 0].astype(np.float64), players_n[..., 1].astype(np.float64))
    tracked = np.isfinite(players_n[..., 0])
    on = tracked & (u >= 0) & (u <= C.IMAGE_W) & (v >= 0) & (v <= C.IMAGE_H)
    n_tr = tracked.sum(1)
    bu, bv = _apply(H, ball_n[:, 0].astype(np.float64), ball_n[:, 1].astype(np.float64))
    b_known = np.isfinite(bu) & np.isfinite(bv)
    b_on = np.full(n, -1, dtype=np.int8)
    b_on[b_known] = ((bu[b_known] >= 0) & (bu[b_known] <= C.IMAGE_W) &
                     (bv[b_known] >= 0) & (bv[b_known] <= C.IMAGE_H)).astype(np.int8)

    return {
        "cam_cx_n": cx.astype(np.float32),
        "cam_cy_n": cy.astype(np.float32),
        "cam_view_width_m": width.astype(np.float32),
        "cam_pan_speed": pan.astype(np.float32),
        "cam_reproj_error": err.astype(np.float32),
        "n_on_screen": on.sum(1).astype(np.int8),
        "frac_tracked_off_screen": np.where(n_tr > 0, 1 - on.sum(1) / np.maximum(n_tr, 1),
                                            np.nan).astype(np.float32),
        "ball_on_screen": b_on,
    }
