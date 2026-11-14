"""Aggregate statistics and the shareable build report (cache/report.txt)."""

import os
import platform

import numpy as np
import pandas as pd

from . import __version__, config as C
from .util import runs


def gap_stats(ball_observed, graded):
    """Length distribution of missing-ball gaps inside graded play."""
    miss = (~ball_observed) & graded
    st, en, val = runs(miss.astype(np.int8))
    lens = (en - st)[val == 1] / C.FPS
    return {
        "n": int(len(lens)),
        "p50_s": float(np.median(lens)) if len(lens) else 0.0,
        "p90_s": float(np.percentile(lens, 90)) if len(lens) else 0.0,
        "max_s": float(lens.max()) if len(lens) else 0.0,
        "share_gt5": float(lens[lens > 5].sum() / max(lens.sum(), 1e-9)),
        "share_gt20": float(lens[lens > 20].sum() / max(lens.sum(), 1e-9)),
    }


def cam_stats(cam, ball_cols, feat, graded):
    """How the camera centre relates to the ball and the two team blocks."""
    gq = graded & np.isfinite(ball_cols["ball_x"]) & np.isfinite(cam["cam_cx"])
    pc = (feat["h_cx_att"] + (C.PITCH_L - feat["a_cx_att"])) / 2
    gp = graded & np.isfinite(pc) & np.isfinite(cam["cam_cx"])
    both = gq & np.isfinite(pc)
    return {
        "corr_x_ball": (float(np.corrcoef(cam["cam_cx"][gq], ball_cols["ball_x"][gq])[0, 1])
                        if gq.sum() > 100 else float("nan")),
        "corr_y_ball": (float(np.corrcoef(cam["cam_cy"][gq], ball_cols["ball_y"][gq])[0, 1])
                        if gq.sum() > 100 else float("nan")),
        "abs_dx_p50": (float(np.median(np.abs(cam["cam_cx"][gq] - ball_cols["ball_x"][gq])))
                       if gq.any() else float("nan")),
        "dy_signed_p50": (float(np.median(cam["cam_cy"][gq] - ball_cols["ball_y"][gq]))
                          if gq.any() else float("nan")),
        "corr_x_players": (float(np.corrcoef(cam["cam_cx"][gp], pc[gp])[0, 1])
                           if gp.sum() > 100 else float("nan")),
        "players_abs_dx_ball_p50": (
            float(np.median(np.abs(pc[both] - ball_cols["ball_x"][both])))
            if both.any() else float("nan")),
    }


def video_time(frame):
    """h:mm:ss on the tactical video."""
    sec = frame / C.FPS
    return "{:d}:{:02d}:{:02d}".format(int(sec // 3600), int(sec % 3600 // 60), int(sec % 60))


def match_clock(frame):
    """mm:ss of the match clock from the published kickoffs (negative before a kickoff)."""
    if frame <= C.HALF_BLOCKS[0][2]:
        t = (frame - C.HALF_BLOCKS[0][1]) / C.FPS
    elif frame < C.KICKOFF_2ND:
        t = (C.KICKOFF_2ND - frame) / C.FPS
        return "2nd-half kickoff -{:02d}:{:02d}".format(int(t // 60), int(t % 60))
    else:
        t = 2700 + (frame - C.KICKOFF_2ND) / C.FPS
    return "{:02d}:{:02d}".format(int(t // 60), int(t % 60))


def stretches(mask, graded, min_len=1, top=3):
    """{frames, stretches, longest: [(len, start)]} of a boolean mask inside graded play."""
    st, en, val = runs((mask & graded).astype(np.int8))
    keep = (val == 1) & ((en - st) >= min_len)
    st, en = st[keep], en[keep]
    return {"frames": int((en - st).sum()), "stretches": int(len(st)),
            "longest": sorted(((int(b - a), int(a)) for a, b in zip(st, en)), reverse=True)[:top]}


def diagnostics(pl, ball_cols, feat, cam, graded):
    """The invariant sweep (docs/PLAN.md): report-only counts, so new kinds of error show up."""
    from .util import out_dist
    X, Y, meas = pl["X"], pl["Y"], pl["meas"]
    with np.errstate(invalid="ignore"):
        off3 = (meas & (out_dist(X, Y) > 3.0)).any(axis=1)
    bx, by = ball_cols["ball_x"].astype(float), ball_cols["ball_y"].astype(float)
    obs = ball_cols["ball_observed"]
    step = np.full(len(bx), np.nan)
    adj = obs[1:] & obs[:-1]
    step[1:] = np.where(adj, np.hypot(np.diff(bx), np.diff(by)) * C.FPS, np.nan)
    med = pd.Series(step).rolling(12, min_periods=3, center=True).median().to_numpy()
    with np.errstate(invalid="ignore"):
        b_out_moving = obs & (out_dist(bx, by) > 3.0) & (med > 5.0)
        cam_jump = np.zeros(len(bx), dtype=bool)
        cam_jump[1:] = np.hypot(np.diff(cam["cam_cx"]), np.diff(cam["cam_cy"])) > 2.0
    return {
        "P4 measured player > 3 m outside the pitch": stretches(off3, graded),
        "B1 ball faster than 42 m/s between consecutive frames": stretches(step > C.BALL_MAX_SPEED, graded),
        "B3 ball > 3 m outside the pitch and moving > 5 m/s": stretches(b_out_moving, graded, 5),
        "F1 camera centre jumps > 2 m in one frame": stretches(cam_jump, graded),
        "F2 homography reprojection error > 2": stretches(cam["cam_reproj_error"] > 2, graded, 5),
    }


def compare_previous(path, df):
    """How much the nearest-team answer changed against the frames table already on disk."""
    if not os.path.isfile(path):
        return None
    try:
        old = pd.read_parquet(path, columns=["closer_team", "graded"])
    except Exception:                                   # noqa: BLE001 -- an old or partial file
        return None
    if len(old) != len(df):
        return None
    g = df["graded"].to_numpy()
    a, b = old["closer_team"].to_numpy(), df["closer_team"].to_numpy()
    either = g & ((a != 0) | (b != 0))
    near = g & (df["d1_min"].to_numpy() < 3)
    return {"frames_with_an_answer": int(either.sum()),
            "changed": int((either & (a != b)).sum()),
            "on_ball_frames": int(near.sum()),
            "on_ball_changed": int((near & (a != b)).sum())}


def write_report(path, ctx):
    L = []
    w = L.append
    w("MTAH 2027 ETL -- build report (aggregate statistics only, safe to share)")
    w("=" * 88)
    w("etl version {}   built in {:.0f} s   python {}   numpy {}   pandas {}".format(
        __version__, ctx["runtime_s"], platform.python_version(), np.__version__, pd.__version__))
    w("")
    w("INPUTS")
    for name, (size, sha) in ctx["inputs"].items():
        w("  {:<38} {:>12,} bytes  sha256 {}".format(name, size, sha[:16]))
    m = ctx["meta"]
    w("")
    w("METADATA")
    w("  teams: home = {!r}, away = {!r}".format(m["home_name"], m["away_name"]))
    w("  slots: {} total, {} home, {} away   (order from {})".format(
        len(m["slot_team"]), int((m["slot_team"] == 0).sum()), int((m["slot_team"] == 1).sum()),
        m["slot_order_source"]))
    w("  player channels: {}   ball channels: {}".format(m["player_channels"], m["ball_channels"]))
    w("  widths seen in .txt: {} slots x {} channels, ball {} channels".format(*ctx["widths"]))
    w("  goalkeeper slots: {} (home {}, away {})".format(
        int(m["slot_is_gk"].sum()),
        [int(i) + 1 for i in np.flatnonzero(m["slot_is_gk"] & (m["slot_team"] == 0))],
        [int(i) + 1 for i in np.flatnonzero(m["slot_is_gk"] & (m["slot_team"] == 1))]))
    w("  ball-state / owner words in XML: {}".format(m["suspect_words"] or "none"))
    w("")
    i = ctx["raw_info"]
    w("RAW PARSE  ({})".format("from cache" if i.get("from_cache") else "parsed now"))
    w("  {:,} lines, frame ids {}..{}, contiguous from 1: {}".format(
        i["n_lines"], i["first_frame_id"], i["last_frame_id"], i["contiguous_from_1"]))
    w("  malformed {}, slow-path {}, out of range {}, duplicates {}".format(
        i["n_bad_lines"], i["n_slow_lines"], i["n_out_of_range"], i["n_duplicate_frames"]))
    w("")
    w("ORIENTATION (home attacks +x after rotation)")
    for h in (1, 2):
        f, mg = ctx["flips"][h]
        w("  half {}: rotated = {:<5}  decision margin {:.1f} m".format(h, str(f), mg))
    df, g = ctx["frames"], ctx["frames"]["graded"].to_numpy()
    bs = ctx["ball_stats"]
    w("")
    w("BALL")
    w("  observed (raw):        {:.1%} of all frames, {:.1%} of graded".format(
        bs["n_observed_raw"] / len(df), ctx["raw_ball_obs_graded"]))
    w("  teleports removed:     {:,}    implausible jumps flagged: {:,}".format(
        bs["n_spikes_removed"], bs["n_jumps_flagged"]))
    w("  short gaps bridged:    {:,} frames (gap <= {} frames)".format(
        bs["n_interpolated"], C.INTERP_MAX_GAP))
    if "second_diff_cm_p50_p90_p99" in bs:
        w("  track character: step speed p50/p90/p99 {:.1f} / {:.1f} / {:.1f} m/s; 2nd difference p50/p90/p99 {:.1f} / {:.1f} / {:.1f} cm".format(
            *bs["step_speed_p50_p90_p99"], *bs["second_diff_cm_p50_p90_p99"]))
        w("     exactly linear steps {:.1%}, not moving {:.1%}   (high 'linear' = the tracker interpolates)".format(
            bs["share_exactly_linear"], bs["share_not_moving"]))
    w("  too-fast runs (> {:.0f} m/s): {:,} runs, median {:.0f} frames; frames flagged unreliable: {:,}".format(
        C.BALL_MAX_SPEED, bs["fast_runs"], bs["fast_run_len_p50"], int(df["ball_unreliable"].sum())))
    if "visible_runs" in bs:
        w("  visible stretches in graded play: {:,}, length p50 / p90 / max {:.1f} / {:.1f} / {:.0f} s".format(
            bs["visible_runs"], *bs["visible_run_s_p50_p90_max"]))
    rn = ctx["runs"]
    if len(rn):
        w("  where stretches start: within 2 m of a touchline {:.0%}, of a goal line {:.0%}, of the centre spot {:.0%}; slow (< 1 m/s) {:.0%}".format(
            (rn["d_touchline_start"] < 2).mean(), (rn["d_goalline_start"] < 2).mean(),
            (rn["d_centre_start"] < 2).mean(), (rn["speed_start"] < 1).mean()))
        w("  where stretches end:   within 2 m of a touchline {:.0%}, of a goal line {:.0%}; already out {:.0%}".format(
            (rn["d_touchline_end"] < 2).mean(), (rn["d_goalline_end"] < 2).mean(),
            (rn["out_dist_end"] > 0).mean()))
    gs = ctx["gap_stats"]
    w("  missing-ball gaps in graded play: {:,} gaps, length p50 / p90 / max {:.1f} / {:.1f} / {:.0f} s".format(
        gs["n"], gs["p50_s"], gs["p90_s"], gs["max_s"]))
    w("  share of missing graded frames inside gaps > 5 s: {:.1%}, > 20 s: {:.1%}".format(
        gs["share_gt5"], gs["share_gt20"]))
    w("  position known after cleaning: {:.1%} of graded frames".format(
        df.loc[g, "ball_x"].notna().mean()))
    w("  out of bounds on graded frames with a position: {:.1%}".format(
        (df.loc[g, "ball_oob"] > 0).sum() / max(1, (df.loc[g, "ball_oob"] >= 0).sum())))
    if "third_channel_present" in bs:
        w("  3rd ball channel: present {:.1%}, corr with ball speed {:.2f}, p50/p99 {}".format(
            bs["third_channel_present"], bs["third_channel_corr_with_speed"],
            tuple(round(v, 2) for v in bs["third_channel_p50_p99"] or ())))
        w("     present on {:,} frames with no ball position; meaning unknown -> kept raw as ball_ch3, not a feature".format(
            bs["third_channel_without_xy"]))
    w("  short gaps NOT bridged (implied speed > {:.0f} m/s): {:,} frames".format(
        C.BALL_MAX_SPEED, bs.get("n_gap_frames_not_bridged", 0)))
    w("")
    w("CAMERA (homography)")
    w("  image centre on pitch known: {:.1%} of graded frames".format(df.loc[g, "cam_cx"].notna().mean()))
    w("  distance camera centre -> ball (graded, ball known): p25/p50/p75 {:.1f} / {:.1f} / {:.1f} m".format(
        *np.nanpercentile(df.loc[g, "cam_d_ball"], [25, 50, 75])))
    cs = ctx["cam_stats"]
    w("  along the pitch (x): corr(camera, ball) {:.2f}, median |dx| {:.1f} m   across (y): corr {:.2f}, median signed dy {:+.1f} m".format(
        cs["corr_x_ball"], cs["abs_dx_p50"], cs["corr_y_ball"], cs["dy_signed_p50"]))
    w("  for comparison, mid-point of the two team blocks: corr(camera x, blocks x) {:.2f}, median |blocks x - ball x| {:.1f} m".format(
        cs["corr_x_players"], cs["players_abs_dx_ball_p50"]))
    w("  view width p50 {:.1f} m   pan speed p50/p95 {:.2f} / {:.2f} m/s".format(
        df.loc[g, "cam_view_width_m"].median(), df.loc[g, "cam_pan_speed"].median(),
        df.loc[g, "cam_pan_speed"].quantile(0.95)))
    w("  tracked players off screen: {:.1%}   ball on screen when known: {:.1%}".format(
        df.loc[g, "frac_tracked_off_screen"].mean(),
        (df.loc[g, "ball_on_screen"] == 1).sum() / max(1, (df.loc[g, "ball_on_screen"] >= 0).sum())))
    w("")
    w("TRACKING (graded frames)")
    w("  objects tracked: mean {:.2f}  (home {:.2f}, away {:.2f})".format(
        df.loc[g, "n_tracked"].mean(), df.loc[g, "h_n_tracked"].mean(),
        df.loc[g, "a_n_tracked"].mean()))
    hs = ctx["hygiene"]
    w("  positions present {:,}, measured (speed present) {:.1%}; the rest are the tracker's guesses".format(
        hs["positions_present"], hs["positions_measured"] / max(1, hs["positions_present"])))
    w("  speed derived from positions for tempo features: {:,} points   duplicate track points removed: {:,}".format(
        hs["speed_derived_from_positions"], hs["duplicate_track_points"]))
    w("  off-pitch points: {:,} just outside (kept for ball distance), {:,} far outside (dropped)".format(
        hs["off_pitch_points"], hs["far_off_points_dropped"]))
    on_h = df["h_n_tracked"] - df["h_n_off_pitch"]
    on_a = df["a_n_tracked"] - df["a_n_off_pitch"]
    w("  frozen frames: {:,}   graded frames with > 11 on the pitch after cleaning: home {:,}, away {:,} (max {} / {})".format(
        hs["frozen_frames"], int((on_h[g] > 11).sum()), int((on_a[g] > 11).sum()),
        int(on_h[g].max()), int(on_a[g].max())))
    w("  collective movement (flow_x) known on {:.1%} of graded frames".format(df.loc[g, "flow_x"].notna().mean()))
    w("  last control team known on {:.1%} of graded frames (home {:.1%} of those)".format(
        (df.loc[g, "last_control_team"] != 0).mean(),
        (df.loc[g, "last_control_team"] == 1).sum() / max(1, (df.loc[g, "last_control_team"] != 0).sum())))
    w("  nearest player to ball: median {:.1f} m (home closer on {:.1%} of frames with a ball)".format(
        df.loc[g, "d1_min"].median(),
        (df.loc[g, "closer_team"] == 1).sum() / max(1, (df.loc[g, "closer_team"] != 0).sum())))
    w("")
    w("CORRECTIONS (docs/PLAN.md)")
    w("  P6 keeper slot given the team defending the goal it stands at: {} frames changed (by slot)".format(
        hs["keeper_team_changed"]))
    w("     unmeasured keeper-slot positions in the opponent's half dropped as ghosts: {}".format(
        hs["keeper_ghosts_dropped"]))
    w("  P9 opposite-team one-object pairs: {:,} track points; frames where the nearest measured player is one (contested, no team): {:,}".format(
        hs["xdup_points"], hs["contested_frames"]))
    w("  D4 contact runs rejected (ball passed without reacting): {:,}".format(hs["contact_runs_rejected_flyby"]))
    ch = ctx.get("change")
    if ch:
        w("  nearest-team answer vs the previous build: changed on {:,} of {:,} graded frames with an answer ({:.1%}); on-ball (< 3 m) {:,} of {:,} ({:.1%})".format(
            ch["changed"], ch["frames_with_an_answer"], ch["changed"] / max(1, ch["frames_with_an_answer"]),
            ch["on_ball_changed"], ch["on_ball_frames"], ch["on_ball_changed"] / max(1, ch["on_ball_frames"])))
    else:
        w("  nearest-team answer vs the previous build: no previous build to compare")
    w("")
    w("SITUATIONS (dead-ball cues, graded frames; longest stretches as video time / match clock)")
    for name, st in ctx["situations"].items():
        w("  {:<18} {:>7,} frames in {:>4} stretches   longest: {}".format(
            name, st["frames"], st["stretches"],
            ", ".join("{} / {} ({:.1f} s)".format(video_time(a), match_clock(a), l / C.FPS)
                      for l, a in st["longest"]) or "-"))
    w("")
    w("DIAGNOSTICS (report only, never corrected; a new non-zero line deserves a look in the video)")
    for name, st in ctx["diagnostics"].items():
        w("  {:<54} {:>6,} frames in {:>3} stretches   longest: {}".format(
            name, st["frames"], st["stretches"],
            ", ".join("{} ({:.1f} s)".format(video_time(a), l / C.FPS) for l, a in st["longest"]) or "-"))
    w("")
    w("FRAMES TABLE")
    w("  {:,} rows x {} columns".format(*df.shape))
    rates = {c: float(df.loc[g, c].isna().mean()) for c in df.columns if df[c].dtype.kind == "f"}
    hi = sorted([(v, k) for k, v in rates.items() if v > 0.5], reverse=True)
    w("  float columns: {}, median NaN rate on graded frames {:.1%}".format(
        len(rates), float(np.median(list(rates.values()))) if rates else 0))
    w("  columns > 50% NaN on graded frames: {}".format(
        ", ".join("{} ({:.0%})".format(k, v) for v, k in hi) or "none"))
    ev = ctx["events"]
    w("")
    w("EVENTS")
    if len(ev):
        tab = ev.groupby(["kind", "half"], observed=True).size().unstack(fill_value=0)
        for kind, row in tab.iterrows():
            w("  {:<10} ".format(kind) + "   ".join("half {}: {:>6,}".format(h, int(v))
                                                    for h, v in row.items()))
        nz = ctx["event_noise"]
        vis_min = ctx["visible_graded_minutes"]
        w("  per minute of graded play with the ball visible ({:.1f} min):".format(vis_min))
        imp = ev[ev["kind"] == "impulse"]
        con = ev[ev["kind"] == "contact"]
        w("    impulses {:.1f}/min (strong {:.1f}/min)   contacts {:.1f}/min (controls >= {} fr: {:.1f}/min)".format(
            len(imp) / vis_min, imp["strong"].sum() / vis_min, len(con) / vis_min, C.CONTROL_MIN,
            con["strong"].sum() / vis_min))
        for k in C.EVENT_SCALES:
            q = nz["k{}".format(k)]
            w("  scale {:>2} fr: dv p50 {:.2f}  p90 {:.2f}  p99 {:.2f}  p99.9 {:.2f} m/s, noise {:.2f} -> impulses at this scale: {:,}".format(
                k, q["dv_median"], *q["dv_p90_p99_p999"], q["dv_noise_scale"],
                int((imp["scale"] == k).sum())))
        w("  impulses on unreliable ball frames: {:,} of {:,}".format(int(imp["at_jump"].sum()), len(imp)))
        if len(con):
            w("  contact length (frames) p50/p90: {:.0f} / {:.0f}   home share {:.1%}".format(
                con["duration"].median(), con["duration"].quantile(0.9), (con["team"] == 0).mean()))
        w("  impulses attributed: home {:.1%}, away {:.1%}, nobody {:.1%}".format(
            (imp["nearest_team"] == 0).mean(), (imp["nearest_team"] == 1).mean(),
            (imp["nearest_team"] == -1).mean()))
    iv = ctx["intervals"]
    w("")
    w("SMART TAGGING")
    w("  {} intervals, {} distinct codes".format(len(iv), iv["code_key"].nunique()))
    for code, n in iv["code"].value_counts().sort_index().items():
        sides = iv.loc[iv["code"] == code, "side"].value_counts().to_dict()
        w("  {:<32} {:>4}   {}".format(code, n, sides))
    w("")
    w("CHECKS")
    for c in ctx["checks"]:
        w("  [{:<4}] {:<58} {}".format(c["status"], c["check"], c["detail"]))
    w("")
    w("OUTPUTS")
    for name, (size, h) in ctx["outputs"].items():
        w("  {:<20} {:>12,} bytes  content {}".format(name, size, h[:16]))
    text = "\n".join(L) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return text
