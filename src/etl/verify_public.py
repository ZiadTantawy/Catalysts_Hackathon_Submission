"""Check what the ETL's signals mean against the public 5 minutes. Aggregates only.

    python -m src.etl.verify_public                 # needs cache/ from src.etl.build
    python -m src.etl.verify_public --gt some.csv   # any frame,label CSV

What this is for (CHALLENGE.md: "the public ground truth is where you can see it"): confirming the
frames are aligned, and seeing what the ground truth means by "dead" and by "owning" -- is the
ball dead from the moment it crosses the line, how long before a restart, does the team nearest
the ball own it, do the tags point the way we assumed.

What this is NOT for: choosing thresholds, windows or models by the numbers printed here. 17
possessions cannot rank anything (EVALUATION.md), and RULES.md forbids using the public window as
a tuning set. Nothing here is written to cache/ and no label ever enters the ETL tables.

The output is aggregate statistics only and is safe to share.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from . import config as C
from .util import runs

DEFAULT_GT = os.path.join(C.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=C.CACHE_DIR)
    ap.add_argument("--gt", default=DEFAULT_GT)
    args = ap.parse_args(argv)

    df = pd.read_parquet(os.path.join(args.cache, "frames.parquet"))
    ev = pd.read_parquet(os.path.join(args.cache, "events.parquet"))
    gt = pd.read_csv(args.gt)
    lab = gt["label"].astype(str).str.strip().str.upper().to_numpy()
    if len(lab) != len(df):
        sys.exit("ground truth has {} rows, frames table {}".format(len(lab), len(df)))
    cov = lab != "X"
    live = (lab == "H") | (lab == "A")
    sign = np.where(lab == "H", 1, np.where(lab == "A", -1, 0)).astype(np.int8)
    P = print

    P("ETL signals against the ground truth -- semantics check, NOT a tuning tool")
    P("=" * 88)
    P("covered frames: {:,}  (H {:,} · A {:,} · D {:,})".format(
        int(cov.sum()), int((lab == "H").sum()), int((lab == "A").sum()), int((lab == "D").sum())))
    f0, f1 = np.flatnonzero(cov)[[0, -1]]
    st, en, val = runs(lab[f0:f1 + 1])
    st, en = st + f0, en + f0
    poss = (val == "H") | (val == "A")
    dead = val == "D"
    P("segments: {} possessions (mean {:.1f} s), {} dead-ball stretches (mean {:.1f} s)".format(
        int(poss.sum()), (en - st)[poss].mean() / C.FPS if poss.any() else 0,
        int(dead.sum()), (en - st)[dead].mean() / C.FPS if dead.any() else 0))
    direct = sum(1 for i in range(1, len(val)) if poss[i] and poss[i - 1])
    P("changes of possession with no stoppage between: {}".format(direct))

    P("")
    P("1. ALIGNMENT  (agreement of 'team nearer the ball' with the owner, as the ETL is shifted)")
    c = df["closer_team"].to_numpy()
    best = None
    line = []
    for lag in (-50, -25, -12, -6, -3, 0, 3, 6, 12, 25, 50):
        cs = np.roll(c, lag)
        m = live & (cs != 0)
        a = float((cs[m] == sign[m]).mean()) if m.any() else float("nan")
        line.append("{:+d}: {:.3f}".format(lag, a))
        if best is None or a > best[1]:
            best = (lag, a)
    P("   " + "   ".join(line))
    P("   best shift {:+d} frames (expect a flat top around 0; a clear peak elsewhere = misalignment)".format(best[0]))

    P("")
    P("2. BALL VISIBILITY by label")
    for L_ in "HAD":
        m = lab == L_
        if m.any():
            P("   {}: ball observed on {:.1%} of frames".format(L_, df.loc[m, "ball_observed"].mean()))

    P("")
    P("3. WHO OWNS THE BALL  (live frames; agreement where the signal has an opinion · coverage)")
    sigs = [("closer_team", "team of the nearest player"),
            ("contact_team", "ball within 1.5 m of a player, now"),
            ("control_team", "contact held >= 0.5 s, now"),
            ("last_contact_team", "last contact, carried forward"),
            ("last_control_team", "last control, carried forward"),
            ("tag_vote_home", "Smart Tagging vote"),
            ("flow_x", "direction everyone is moving"),
            ("flow_x__c3s", "same, 3 s average")]
    for col, what in sigs:
        if col not in df:
            continue
        v = np.sign(np.nan_to_num(df[col].to_numpy(dtype=np.float64)))
        m = live & (v != 0)
        P("   {:<20} {:<36} agree {:.1%} · covers {:.1%}".format(
            col, what, (v[m] == sign[m]).mean() if m.any() else float("nan"),
            m.sum() / max(1, live.sum())))
    for code in ("tag_build_up", "tag_progression", "tag_ball_in_final_third",
                 "tag_attacking_transition", "tag_long_ball", "tag_set_pieces"):
        t = df[code + "_team"].to_numpy()
        m = live & (df[code].to_numpy() == 1) & (t != 0)
        if m.sum() > 50:
            P("   {:<28} tagged team owns the ball on {:.1%} of {:,} frames".format(
                code, (t[m] == sign[m]).mean(), int(m.sum())))

    P("")
    P("4. WHAT DEAD LOOKS LIKE  (medians / shares by label)")
    cols = [("all_mean_speed", "median"), ("n_tracked", "median"), ("cam_pan_speed", "median"),
            ("ball_speed", "median"), ("kickoff_shape", "median"), ("n_off_pitch", "mean"),
            ("flow_x", "absmedian")]
    P("   {:<18}".format("") + "".join("{:>10}".format(L_) for L_ in "HAD"))
    for col, how in cols:
        out = []
        for L_ in "HAD":
            x = df.loc[lab == L_, col].astype(float)
            if how == "median":
                out.append(x.median())
            elif how == "mean":
                out.append(x.mean())
            else:
                out.append(x.abs().median())
        P("   {:<18}".format(col) + "".join("{:>10.2f}".format(v) for v in out))
    for name, mask in (("ball out of bounds", df["ball_oob"].to_numpy() > 0),
                       ("ball stationary", df["ball_stationary"].to_numpy() == 1),
                       ("set-piece tag active", df["tag_set_pieces"].to_numpy() == 1)):
        P("   {:<18}".format(name) + "".join(
            "{:>10.1%}".format(mask[lab == L_].mean() if (lab == L_).any() else float("nan"))
            for L_ in "HAD"))

    P("")
    P("5. STOPPAGES  ({} in the window)".format(int(dead.sum())))
    if dead.any():
        ds, de = st[dead], en[dead]
        obs = df["ball_observed"].to_numpy()
        oob = df["ball_oob"].to_numpy()
        out_before = np.mean([np.any(oob[max(0, a - 50):a + 25] > 0) for a in ds])
        seen_at_start = np.mean([obs[a] for a in ds])
        seen_at_end = np.mean([obs[min(b, len(obs) - 1)] for b in de])
        P("   ball seen out of bounds within 2 s before / 1 s after the stoppage starts: {:.0%}".format(out_before))
        P("   ball observed at the first dead frame: {:.0%}   at the first live frame after: {:.0%}".format(
            seen_at_start, seen_at_end))
        sp = np.flatnonzero((df["frames_since_setpiece_start"].to_numpy() == 0))
        if len(sp):
            dt = [(sp[np.argmin(np.abs(sp - b))] - b) / C.FPS for b in de]
            P("   nearest set-piece tag start relative to the restart: median {:+.1f} s, p10 {:+.1f}, p90 {:+.1f}"
              "   (negative = tag starts before play resumes)".format(
                  np.median(dt), np.percentile(dt, 10), np.percentile(dt, 90)))
        gap = df["ball_frames_since_seen"].to_numpy()
        P("   median time since the ball was last seen, at the restart: {:.1f} s".format(
            np.median([max(gap[min(b, len(gap) - 1)], 0) for b in de]) / C.FPS))

    P("")
    P("6. CHANGES OF POSSESSION IN OPEN PLAY  ({} in the window)".format(direct))
    if direct:
        tr = np.array([st[i] for i in range(1, len(val)) if poss[i] and poss[i - 1]])
        imp = ev.loc[ev["kind"] == "impulse", "frame"].to_numpy()
        con = ev.loc[ev["kind"] == "contact", ["frame", "team"]].to_numpy()
        near_imp = np.mean([np.any(np.abs(imp - t) <= 12) for t in tr]) if len(imp) else 0.0
        hit = []
        for t in tr:
            new = 0 if lab[t] == "H" else 1
            m = (con[:, 0] >= t - 25) & (con[:, 0] <= t + 25) & (con[:, 1] == new) if len(con) else []
            hit.append(bool(np.any(m)))
        P("   an impulse within 0.5 s of the change: {:.0%}".format(near_imp))
        P("   a contact by the NEW team starting within 1 s of the change: {:.0%}".format(np.mean(hit)))
        P("   ball observed at the change: {:.0%}".format(df["ball_observed"].to_numpy()[tr].mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
