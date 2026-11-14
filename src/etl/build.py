"""ETL: build the frame-level tables from the raw tactical bundle. One command, deterministic.

    python -m src.etl.build                      # data from ./data or $MTAH_DATA_DIR
    python -m src.etl.build --out cache          # where to write (default ./cache)
    python -m src.etl.build --force-parse        # ignore the cached parse of the .txt

Writes (all gitignored, never to be committed or shared -- frame-level exports reconstruct the
licensed data):

    cache/frames.parquet      199,319 rows, one per video frame = submission row
    cache/events.parquet      ball events (impulses, contacts, exits/entries, lost/found)
    cache/ball_runs.parquet   one row per continuous stretch of observed ball
    cache/tagging.parquet     Smart Tagging intervals in frames, team mapped to H / A / N/A
    cache/players_m.npy       [199319, 46, C] oriented metres (home attacks +x), as delivered
    cache/players_meas.npy    [199319, 46] bool, position measured (speed present), not guessed
    cache/players_team.npy    [199319, 46] int8 per-frame team after corrections (-1 = dropped)
    cache/players_keeper.npy  [199319, 46] bool, per-frame goalkeeper role
    cache/ball_m.npy          [199319, Cb] oriented metres, raw (before cleaning)
    cache/slot_team.npy       [46] 0 = home, 1 = away
    cache/manifest.json       inputs' sha256, every parameter, output content hashes
    cache/report.txt          aggregate statistics and checks ONLY -- the one file safe to share

Exit status is non-zero if any FAIL-level check fails.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

from . import (__version__, ball, checks, config as C, features, io_atd, io_homography,
               io_tagging, orient, players, report, situations)
from games import require  # noqa: E402  (config puts the repo root on sys.path)

PARAM_NAMES = [k for k in dir(C) if k.isupper() and k not in ("ROOT", "CACHE_DIR")]


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(df):
    """Hash of the table's values and column names, independent of parquet writer metadata."""
    h = hashlib.sha256()
    for col in df.columns:
        h.update(col.encode())
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype)
        if not numeric or isinstance(s.dtype, pd.CategoricalDtype):
            h.update("\x1f".join(str(v) for v in s.tolist()).encode())
        else:
            a = s.to_numpy(dtype=np.float64, na_value=np.nan) if s.hasnans else s.to_numpy()
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def load_raw(raw_path, raw_sha, out_dir, force):
    cache = os.path.join(out_dir, "raw_parse.npz")
    n_slots, n_pc, n_bc = io_atd.sniff_widths(raw_path)
    if not force and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z["raw_sha"]) == raw_sha:
            info = json.loads(str(z["info"]))
            info["from_cache"] = True
            return z["players"], z["ball"], info, (n_slots, n_pc, n_bc)
    players, ball_raw, info = io_atd.read_raw(raw_path, n_slots, n_pc, n_bc)
    info["from_cache"] = False
    np.savez(cache, players=players, ball=ball_raw, raw_sha=np.array(raw_sha),
             info=np.array(json.dumps(info)))
    return players, ball_raw, info, (n_slots, n_pc, n_bc)


def build(data_dir=None, out_dir=C.CACHE_DIR, force_parse=False, quiet=False):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    meta_path, raw_path, tag_path, hom_path = require(
        ["atd_meta", "atd_raw", "smart_tagging", "homography"], data_dir=data_dir)
    inputs = {os.path.basename(p): (os.path.getsize(p), sha256_file(p))
              for p in (meta_path, raw_path, tag_path, hom_path)}
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))

    say("[1/8] metadata + raw tracking")
    meta = io_atd.read_meta(meta_path)
    raw_players, ball_raw, raw_info, widths = load_raw(
        raw_path, inputs[os.path.basename(raw_path)][1], out_dir, force_parse)
    if widths[0] != len(meta["slot_team"]):
        raise ValueError("raw file has {} slots, metadata {}".format(widths[0], len(meta["slot_team"])))
    slot_team = meta["slot_team"]

    say("[2/8] camera (homography)")
    H, hvalid, herr = io_homography.load(hom_path)
    cam = io_homography.camera_features(H, hvalid, herr, raw_players, ball_raw)
    del H

    say("[3/8] timeline + orientation")
    tl = orient.timeline()
    p_m, b_m = orient.to_metres(raw_players, ball_raw)
    flips = orient.detect_flips(p_m, slot_team, tl["half"])
    p_o, b_o, flip_frame = orient.apply_orientation(p_m, b_m, tl["phase"], flips)
    del p_m, b_m
    cx_m = cam.pop("cam_cx_n") * C.PITCH_L
    cy_m = cam.pop("cam_cy_n") * C.PITCH_W
    cam["cam_cx"] = np.where(flip_frame, C.PITCH_L - cx_m, cx_m).astype(np.float32)
    cam["cam_cy"] = np.where(flip_frame, C.PITCH_W - cy_m, cy_m).astype(np.float32)

    say("[4/8] ball cleaning")
    ball_cols, ball_stats = ball.clean(b_o, tl["half"])
    raw_obs = np.isfinite(ball_raw[:, 0]) & np.isfinite(ball_raw[:, 1])
    raw_obs_graded = float(raw_obs[tl["graded"]].mean())
    cam["cam_d_ball"] = np.hypot(cam["cam_cx"] - ball_cols["ball_x"],
                                 cam["cam_cy"] - ball_cols["ball_y"]).astype(np.float32)

    say("[5/8] players, team, contest and dead-ball features")
    pl, pl_stats = players.prepare(p_o, slot_team, meta["slot_is_gk"])
    feat, near, hyg_stats, _ = features.team_contest_deadball(p_o, pl, ball_cols)
    hyg_stats.update(pl_stats)
    con_cols, con_episodes, con_stats = features.contacts(feat, tl["half"], ball_cols)
    feat.update(con_cols)
    hyg_stats.update(con_stats)
    sit_cols, sit_stats = situations.compute(pl, feat, ball_cols, tl)
    diag = report.diagnostics(pl, ball_cols, feat, cam, tl["graded"])

    say("[6/8] smart tagging")
    iv = io_tagging.load(tag_path, meta["home_name"], meta["away_name"])
    tag_cols, unknown_codes = io_tagging.per_frame(iv)

    say("[7/8] assemble + events + context windows")
    df = pd.DataFrame({"frame": np.arange(C.N_FRAMES, dtype=np.int32), **tl,
                       "rotated": flip_frame, **ball_cols, **feat, **sit_cols, **cam, **tag_cols})
    df = pd.concat([df, features.context_windows(df)], axis=1)
    ev, ev_noise = ball.events(ball_cols, tl["half"], near, con_episodes)
    runs_df, run_id = ball.runs_table(ball_cols, tl["half"])
    df["ball_run_id"] = run_id

    say("[8/8] write + check")
    change = report.compare_previous(os.path.join(out_dir, "frames.parquet"), df)
    outs = {
        "frames.parquet": df,
        "events.parquet": ev,
        "ball_runs.parquet": runs_df,
        "tagging.parquet": iv,
    }
    outputs = {}
    for name, table in outs.items():
        path = os.path.join(out_dir, name)
        table.to_parquet(path, index=False)
        outputs[name] = (os.path.getsize(path), content_hash(table))
    for name, arr in (("players_m.npy", p_o), ("players_meas.npy", pl["meas"]),
                      ("players_team.npy", pl["team"]), ("players_keeper.npy", pl["keeper"]),
                      ("ball_m.npy", b_o), ("slot_team.npy", slot_team)):
        path = os.path.join(out_dir, name)
        np.save(path, arr)
        outputs[name] = (os.path.getsize(path), hashlib.sha256(arr.tobytes()).hexdigest())

    obs_g = ball_cols["ball_observed"] & tl["graded"]
    ctx = {
        "cam_stats": report.cam_stats(cam, ball_cols, feat, tl["graded"]),
        "runs": runs_df,
        "gap_stats": report.gap_stats(ball_cols["ball_observed"], tl["graded"]),
        "visible_graded_minutes": float(obs_g.sum() / C.FPS / 60),
        "raw_info": raw_info, "meta": meta, "frames": df, "intervals": iv, "events": ev,
        "flips": flips, "ball_stats": ball_stats, "raw_ball_obs_graded": raw_obs_graded,
        "unknown_codes": unknown_codes, "inputs": inputs, "widths": widths,
        "outputs": outputs, "hygiene": hyg_stats, "event_noise": ev_noise,
        "situations": sit_stats, "diagnostics": diag, "change": change,
    }
    ctx["checks"] = checks.run(ctx)
    ctx["runtime_s"] = time.time() - t0

    manifest = {
        "etl_version": __version__,
        "inputs": {k: {"bytes": s, "sha256": h} for k, (s, h) in inputs.items()},
        "outputs": {k: {"bytes": s, "content_sha256": h} for k, (s, h) in outputs.items()},
        "params": {k: getattr(C, k) for k in PARAM_NAMES
                   if isinstance(getattr(C, k), (int, float, str, dict, list, tuple))},
        "orientation": {str(h): {"rotated": f, "margin_m": round(mg, 2)}
                        for h, (f, mg) in flips.items()},
        "columns": {"frames": list(df.columns), "events": list(ev.columns),
                    "ball_runs": list(runs_df.columns),
                    "tagging": list(iv.columns)},
        "checks": ctx["checks"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
    text = report.write_report(os.path.join(out_dir, "report.txt"), ctx)
    say("")
    say(text)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None, help="bundle root (default ./data or $MTAH_DATA_DIR)")
    ap.add_argument("--out", default=C.CACHE_DIR, help="output folder (default ./cache)")
    ap.add_argument("--force-parse", action="store_true", help="re-parse the .txt even if cached")
    args = ap.parse_args(argv)
    ctx = build(args.data_dir, args.out, args.force_parse)
    failed = [c for c in ctx["checks"] if c["status"] == "FAIL"]
    if failed:
        print("{} FAIL-level check(s) -- do not build on this ETL output until fixed.".format(len(failed)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
