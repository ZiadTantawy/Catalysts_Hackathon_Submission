"""Stage 11: invariants the ETL output must satisfy, checked on every build.

FAIL  a contract is broken (row count, frame index, graded-frame count); the build exits non-zero.
WARN  a number differs from what DATA.md publishes for the real match, or a heuristic is weak.
      On the synthetic test bundle some WARNs are expected; on the real data each one deserves a
      look before anything is built on top.
PASS  as expected.
"""

import numpy as np

from . import config as C


def _row(name, ok, detail, level="WARN"):
    return {"check": name, "status": "PASS" if ok else level, "detail": detail}


def run(ctx):
    r = []
    info, meta, df, iv, ev, flips = (ctx["raw_info"], ctx["meta"], ctx["frames"], ctx["intervals"],
                                     ctx["events"], ctx["flips"])
    graded = df["graded"].to_numpy()

    r.append(_row("raw: one record per video frame, ids 1..N", info["contiguous_from_1"],
                  "{} lines, ids {}..{}, {} covered, {} duplicates".format(
                      info["n_lines"], info["first_frame_id"], info["last_frame_id"],
                      info["n_frames_covered"], info["n_duplicate_frames"]), "FAIL"))
    r.append(_row("raw: no malformed lines", info["n_bad_lines"] == 0 and info["n_slow_lines"] == 0,
                  "{} malformed, {} parsed slowly".format(info["n_bad_lines"], info["n_slow_lines"])))
    r.append(_row("frames table: N rows, frame == row", len(df) == C.N_FRAMES and
                  np.array_equal(df["frame"].to_numpy(), np.arange(C.N_FRAMES)),
                  "{} rows".format(len(df)), "FAIL"))
    r.append(_row("graded frames == 158857", int(graded.sum()) == 158857,
                  "{} graded".format(int(graded.sum())), "FAIL"))
    r.append(_row("metadata: 46 track slots", len(meta["slot_team"]) == C.EXPECTED["n_slots"],
                  "{} slots ({} home / {} away)".format(len(meta["slot_team"]),
                                                       int((meta["slot_team"] == 0).sum()),
                                                       int((meta["slot_team"] == 1).sum()))))
    r.append(_row("metadata: slot order consistent", meta["slot_order_agrees_with_player_list"],
                  "order taken from " + meta["slot_order_source"]))
    r.append(_row("metadata: no ball-state / owner channel", not meta["suspect_words"],
                  "words found: {}".format(meta["suspect_words"] or "none")))

    obs = df["ball_observed"].to_numpy()
    raw_obs_all = ctx["ball_stats"]["n_observed_raw"] / C.N_FRAMES
    raw_obs_graded = ctx["raw_ball_obs_graded"]
    r.append(_row("ball visible ~56.9% of all frames (raw)",
                  abs(raw_obs_all - C.EXPECTED["ball_visible_all"]) < 0.01,
                  "{:.1%}".format(raw_obs_all)))
    r.append(_row("ball visible ~59.3% of graded frames (raw)",
                  abs(raw_obs_graded - C.EXPECTED["ball_visible_graded"]) < 0.01,
                  "{:.1%}".format(raw_obs_graded)))
    tracked = df.loc[graded, "n_tracked"].mean()
    r.append(_row("~20.5 objects tracked per graded frame",
                  abs(tracked - C.EXPECTED["tracked_mean_graded"]) < 0.5, "{:.2f}".format(tracked)))

    r.append(_row("orientation: home switches ends at halftime", flips[1][0] != flips[2][0],
                  "half1 flip={} (margin {:.1f} m), half2 flip={} (margin {:.1f} m)".format(
                      flips[1][0], flips[1][1], flips[2][0], flips[2][1])))
    r.append(_row("orientation: confident in both halves", min(flips[1][1], flips[2][1]) > 5.0,
                  "smallest margin {:.1f} m (want > 5)".format(min(flips[1][1], flips[2][1]))))
    # after orientation, home's centroid should sit at lower x than away's on most graded frames
    hx = df.loc[graded, "h_cx_att"].to_numpy()
    ax = C.PITCH_L - df.loc[graded, "a_cx_att"].to_numpy()
    ok = np.isfinite(hx) & np.isfinite(ax)
    frac = float(np.mean(hx[ok] < ax[ok])) if ok.any() else float("nan")
    r.append(_row("orientation: home block deeper than away block on most frames",
                  frac > 0.5, "{:.1%} of graded frames".format(frac)))

    for half, truth in C.EXPECTED["kickoffs"].items():
        sp = iv[(iv["code_key"] == C.SET_PIECE_CODE) & (iv["type"].str.lower() == "kick off (start)")
                & (iv["half_tag"] == half)]
        hit = bool(((sp["start_frame"] <= truth) & (sp["end_frame"] >= truth)).any())
        r.append(_row("tagging: {} kickoff tag contains frame {}".format(half, truth), hit,
                      "{} matching tags".format(len(sp))))
    unm = int((iv["side"] == "unmapped").sum())
    r.append(_row("tagging: every Team value mapped", unm == 0,
                  "{} unmapped; values seen: {}".format(
                      unm, sorted(iv["side"].value_counts().to_dict().items()))))
    r.append(_row("tagging: only known codes", not ctx["unknown_codes"],
                  "unknown: {}".format(ctx["unknown_codes"] or "none")))

    vis = ctx["visible_graded_minutes"]
    if len(ev):
        n_imp = int((ev["kind"] == "impulse").sum())
        n_con = int((ev["kind"] == "contact").sum())
    else:
        n_imp = n_con = 0
    # football prior: a team side makes roughly 8-15 passes a minute in total, plus receptions,
    # dribbles, clearances and deflections; measured over time the ball is actually visible
    r.append(_row("events: impulse rate 3-60 per visible-ball minute",
                  3 <= n_imp / max(vis, 1e-9) <= 60,
                  "{} impulses, {:.1f}/min over {:.0f} visible min".format(n_imp, n_imp / max(vis, 1e-9), vis)))
    r.append(_row("events: contact rate 3-60 per visible-ball minute",
                  3 <= n_con / max(vis, 1e-9) <= 60,
                  "{} contacts, {:.1f}/min".format(n_con, n_con / max(vis, 1e-9))))
    gk = meta["slot_is_gk"]
    r.append(_row("metadata: one goalkeeper slot per team",
                  int((gk & (meta["slot_team"] == 0)).sum()) == 1 and
                  int((gk & (meta["slot_team"] == 1)).sum()) == 1,
                  "{} GK slots found".format(int(gk.sum()))))
    hs = ctx["hygiene"]
    share = hs["positions_measured"] / max(1, hs["positions_present"])
    r.append(_row("players: most positions measured (speed present)", share > 0.8,
                  "{:.1%} measured".format(share)))
    kd = df["known_dead"].to_numpy()
    r.append(_row("timeline: 2nd half graded before kickoff = 2934 frames (FAQ.md)",
                  int(kd.sum()) == 2934 and bool(graded[kd].all()),
                  "{} frames".format(int(kd.sum())), "FAIL"))
    for col in ("contact_team", "control_team"):
        bad = int(((df[col].to_numpy() != 0) & ~graded).sum())
        r.append(_row("contacts only inside graded play: " + col, bad == 0,
                      "{} frames outside".format(bad)))
    cx = ctx["cam_stats"]["corr_x_ball"]
    r.append(_row("camera: follows the ball along the pitch (corr > 0.5)", cx > 0.5,
                  "corr {:.2f}, median |dx| {:.1f} m".format(cx, ctx["cam_stats"]["abs_dx_p50"])))
    tl_nan = int(df[["half", "phase", "graded"]].isna().sum().sum())
    r.append(_row("timeline columns complete", tl_nan == 0, "{} NaN".format(tl_nan), "FAIL"))
    return r
