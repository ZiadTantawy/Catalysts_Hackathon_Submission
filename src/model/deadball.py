"""Dead-ball detector: for every graded frame, is the ball dead (D) or live?

    python -m src.model.deadball            # reads cache/, writes cache/deadball.parquet + report

Rules are the ones agreed in docs/PLAN.md ("Dead-ball detector: rules"). Each situation adds evidence
per frame (positive = dead, negative = live); a two-state decoder then picks, per half, the labelling
that maximises total evidence minus SWITCH_COST for every change of state. A stoppage therefore needs
several seconds of consistent evidence to start, and once started it lasts until the evidence for live
play outweighs it -- a stoppage lasts until play restarts.

Every weight below is a physical or footballing prior chosen before looking at any result. None was
fitted to the public ground truth or to anything a person labelled on video (RULES.md). Weights are in
"evidence per frame"; at 25 fps, 0.1 per frame sustained for 1 s is 2.5.

Output: cache/deadball.parquet (frame, dead_score, dead: 1 dead / 0 live / -1 ungraded) and
cache/deadball_report.txt (aggregate statistics and timestamps to check; safe to share).
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

from ..etl import config as EC
from ..etl.report import match_clock, video_time
from ..etl.util import runs, runs_true

FPS = EC.FPS
L, W = EC.PITCH_L, EC.PITCH_W

# ---- decoder ------------------------------------------------------------------------------------
SWITCH_COST = 8.0            # evidence needed to change state once; a stoppage needs ~2x this in total

# ---- live play ----------------------------------------------------------------------------------
PLAYERS_MID = 1.4            # m/s, players' 3 s mean speed: neutral point between play and stoppage
PLAYERS_SPAN = 0.6           # m/s either side reaches full weight
W_PLAYERS = 0.15             # slow players -> dead, fast players -> live
BALL_MOVING = 4.0            # m/s, ball speed (0.5 s median) that only happens in play ...
W_BALL_MOVING = -0.25        # ... while the ball is inside the pitch
W_CONTACT = -0.15            # a measured player touching a ball that reacts
W_UNSEEN = 0.03              # ball not seen: weak, it is missing in play too

# ---- dead ball ----------------------------------------------------------------------------------
OUT_CLEAR = 1.0              # m beyond a line: clearly out
W_OUT_CLEAR = 0.5
OUT_EDGE = 0.5               # m beyond a line: probably out (tracking error ~0.5 m)
W_OUT_EDGE = 0.2
W_STILL = 0.3                # ball stationary for seconds (restart being set up)
W_ISOLATED = 0.3             # ball slow and far from every measured player
W_GATHER = 0.5               # crowd, set-piece crowd, cooling break
W_BLACKOUT = 0.3             # measured players collapse
W_SURPLUS = 0.05             # > 11 of one team: weak, it also happens in play
LOST_LINE_DIST = 3.0         # m: ball last seen this close to a line ...
LOST_LINE_SPEED = 2.0        # ... moving towards it at least this fast ...
LOST_MIN = 2 * FPS           # ... and unseen for at least 2 s: probably out
W_LOST_LINE = 0.3
# A goal (inside a GOALS tag, which starts at the attack): the first moment the ball is seen behind
# the goal line near the goal (the net), or is last seen inside a penalty box and then lost for
# GOAL_UNSEEN. Dead from there until the next kickoff tag.
GOAL_NET_HALF_WIDTH = 9.0    # m from the goal's centre line, behind the goal line (posts +-3.66, + error)
GOAL_UNSEEN = 3 * FPS
GOAL_KICKOFF_MAX = 6 * 60 * FPS   # the kickoff that ends it must follow within 6 minutes
W_GOAL = 0.6
# Ball over a line (an exit event): dead from the crossing until the ball comes back onto the pitch
# (throw-in, or the ball placed for a goal kick / corner), at most OUT_HOLD_MAX. A spare ball or a
# ball rolled back after it went out does not make play live.
OUT_HOLD_MAX = 90 * FPS
W_OUT_HOLD = 0.4
# Foul / offside / injury (rule 4): a SUDDEN collective slowdown of the measured players (guessed
# positions are frozen and would fake one when the camera pans) ...
SLOW_PRE = 2 * FPS           # mean speed over the previous 2 s ...
SLOW_POST = 2 * FPS          # ... against the next 2 s
SLOW_FROM = 2.0              # m/s: players were running ...
SLOW_TO = 1.2                # m/s: ... and now are not ...
SLOW_DROP = 0.8              # m/s: ... a drop of at least this
# ... is only a candidate. It becomes a stoppage if what follows confirms it within CONFIRM: players
# stay slow for CONFIRM_SLOW (or the ball is set, a crowd forms, the ball is lost near a line), and
# the ball is NOT moving between players for more than CONFIRM_LIVE_MAX (a team settling into
# possession keeps passing).
CONFIRM = 8 * FPS
CONFIRM_SLOW = 4 * FPS
CONFIRM_LIVE_MAX = FPS
# A confirmed stoppage is held until play restarts: everyone accelerates together, or the ball moves
# inside the pitch with a touch for RESTART_BALL, or the ball comes back in, or a kickoff; at most
# STOP_HOLD_MAX. Clear live evidence still outweighs the hold inside the decoder.
RESTART_FROM = 1.2           # m/s: players' mean speed over the previous 2 s at most this ...
RESTART_TO = 1.6             # ... and over the next 2 s at least this (between the typical speed of
                             # a stoppage, ~1.2, and of play, ~2)
RESTART_WIN = 2 * FPS
RESTART_BALL = FPS           # ball moving inside the pitch for 1 s, with a touch within 1 s of it
STOP_HOLD_MAX = 90 * FPS
W_STOP_HOLD = 0.4


def _rolling(a, w, how="mean", min_periods=None):
    s = pd.Series(a)
    r = s.rolling(w, center=True, min_periods=min_periods or max(1, w // 3))
    return (r.mean() if how == "mean" else r.median()).to_numpy()


def _backward_velocity(x, y, obs, k=5):
    """Velocity at each observed frame from the position k frames earlier (m/s); NaN otherwise."""
    vx = np.full(len(x), np.nan)
    vy = np.full(len(x), np.nan)
    ok = obs[k:] & obs[:-k]
    vx[k:] = np.where(ok, (x[k:] - x[:-k]) * FPS / k, np.nan)
    vy[k:] = np.where(ok, (y[k:] - y[:-k]) * FPS / k, np.nan)
    return vx, vy


def _lost_near_line(df, x, y, obs, vx, vy):
    """Frames inside a gap that started with the ball near a line and moving towards it."""
    n = len(df)
    out = np.zeros(n, dtype=bool)
    st, en = runs_true(~obs & df["graded"].to_numpy())
    for a, b in zip(st, en):
        p = a - 1
        if p < 0 or not obs[p] or b - a < LOST_MIN:
            continue
        toward = [
            (y[p] <= LOST_LINE_DIST and vy[p] < -LOST_LINE_SPEED),
            (W - y[p] <= LOST_LINE_DIST and vy[p] > LOST_LINE_SPEED),
            (x[p] <= LOST_LINE_DIST and vx[p] < -LOST_LINE_SPEED),
            (L - x[p] <= LOST_LINE_DIST and vx[p] > LOST_LINE_SPEED),
        ]
        if any(toward):
            out[a:b] = True
    return out


def _goals(df, tg, x, y, obs):
    """[(goal_frame, kickoff_frame)] for goal tags the data agrees with; and the frame mask."""
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    found = []
    kick = tg[(tg["code_key"] == "set pieces") & tg["type"].str.lower().str.startswith("kick off")]
    dy = np.abs(y - W / 2)
    in_net = obs & ((x < 0) | (x > L)) & (dy <= GOAL_NET_HALF_WIDTH)
    in_box = obs & ((x <= EC.BOX_DEPTH) | (L - x <= EC.BOX_DEPTH)) & (dy <= EC.BOX_HALF_WIDTH)
    for r in tg[tg["code_key"] == "goals"].itertuples():
        a, b = max(r.start_frame, 0), min(r.end_frame, n - 1)
        hit = None
        for t in range(a, b):
            if in_net[t]:
                hit = t
                break
            if in_box[t] and not obs[t + 1:t + 1 + GOAL_UNSEEN].any():
                hit = t + 1
                break
        if hit is None:
            continue
        ko = kick[(kick["start_frame"] > hit) & (kick["start_frame"] <= hit + GOAL_KICKOFF_MAX)]
        if ko.empty:
            continue
        k = int(ko["start_frame"].iloc[0])
        mask[hit:k] = True
        found.append((hit, k))
    return found, mask


def measured_speed(cache):
    """Mean tracker speed of the measured, kept players per frame (m/s); NaN with nobody measured."""
    p = np.load(os.path.join(cache, "players_m.npy"), mmap_mode="r")
    meas = np.load(os.path.join(cache, "players_meas.npy"))
    team = np.load(os.path.join(cache, "players_team.npy"))
    sp = np.where(meas & (team >= 0), p[..., 2], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(sp, axis=1)


def _forward_mean(a, w):
    return pd.Series(a[::-1]).rolling(w, min_periods=max(1, w // 2)).mean().to_numpy()[::-1]


def restart_picture(df, ev, tg, speed, ball_moving, contact):
    """Frames where play visibly restarts: everyone accelerates together, the ball moves inside the
    pitch with a touch, the ball comes back onto the pitch, or a kickoff tag starts."""
    n = len(df)
    before = pd.Series(speed).rolling(RESTART_WIN, min_periods=RESTART_WIN // 2).mean().to_numpy()
    after = _forward_mean(speed, RESTART_WIN)
    accel = (np.r_[np.nan, before[:-1]] <= RESTART_FROM) & (after >= RESTART_TO)
    moving = _forward_mean(ball_moving.astype(float), RESTART_BALL) >= 0.99      # the next 1 s
    touch_near = pd.Series(contact).rolling(2 * RESTART_BALL + 1, center=True, min_periods=1) \
        .max().to_numpy() > 0
    moving_touch = moving & touch_near
    marks = np.zeros(n, dtype=bool)
    marks[ev.loc[ev["kind"] == "entry", "frame"].to_numpy()] = True
    kick = tg[(tg["code_key"] == "set pieces") & tg["type"].str.lower().str.startswith("kick off")]
    marks[np.clip(kick["start_frame"].to_numpy(), 0, n - 1)] = True
    return accel | moving_touch | marks


def _hold(n, start, restart, max_len, min_len=FPS):
    """Frames from `start` until the first restart picture after min_len, at most max_len."""
    later = np.flatnonzero(restart[start + min_len:min(n, start + max_len)])
    return start + min_len + int(later[0]) if len(later) else min(n, start + max_len)


def _stoppages_from_slowdown(df, speed, ball_moving, other_dead, restart):
    """Confirmed sudden collective slowdowns, each held until play restarts. Returns (mask, onsets)."""
    n = len(df)
    graded = df["graded"].to_numpy()
    pre = pd.Series(speed).rolling(SLOW_PRE, min_periods=SLOW_PRE // 2).mean().to_numpy()
    post = _forward_mean(speed, SLOW_POST)
    pre_prev = np.r_[np.nan, pre[:-1]]
    cand = graded & (pre_prev >= SLOW_FROM) & (post <= SLOW_TO) & (pre_prev - post >= SLOW_DROP)
    st, _ = runs_true(cand)
    slow3 = _rolling(speed, 3 * FPS) < SLOW_TO

    mask = np.zeros(n, dtype=bool)
    onsets = []
    for t in st:
        w = slice(t, min(n, t + CONFIRM))
        if ball_moving[w].sum() > CONFIRM_LIVE_MAX:
            continue                                              # play carried on
        if slow3[w].sum() < CONFIRM_SLOW and not other_dead[w].any():
            continue                                              # not confirmed
        mask[t:_hold(n, t, restart, STOP_HOLD_MAX)] = True
        onsets.append(int(t))
    return mask & graded, onsets


def evidence(df, ev, tg, speed):
    """Per-frame dead-ball evidence and its named components. speed: measured players' mean speed."""
    obs = df["ball_observed"].to_numpy()
    x = df["ball_x"].to_numpy(dtype=float)
    y = df["ball_y"].to_numpy(dtype=float)
    vx, vy = _backward_velocity(x, y, obs)

    players = _rolling(speed, 3 * FPS)
    ball_speed = _rolling(np.where(obs, df["ball_speed"].to_numpy(dtype=float), np.nan),
                          FPS // 2, how="median")
    out_dist = np.nan_to_num(df["ball_out_dist"].to_numpy(dtype=float), nan=0.0)
    inside = obs & (out_dist == 0)

    comp = {}
    comp["players"] = W_PLAYERS * np.clip((PLAYERS_MID - np.nan_to_num(players, nan=PLAYERS_MID))
                                          / PLAYERS_SPAN, -1, 1)
    comp["ball_moving"] = np.where(inside & (np.nan_to_num(ball_speed) > BALL_MOVING), W_BALL_MOVING, 0)
    comp["contact"] = np.where(df["contact_team"].to_numpy() != 0, W_CONTACT, 0)
    comp["unseen"] = np.where(~obs, W_UNSEEN, 0)
    comp["ball_out"] = np.where(obs & (out_dist > OUT_CLEAR), W_OUT_CLEAR,
                                np.where(obs & (out_dist > OUT_EDGE), W_OUT_EDGE, 0))
    comp["still"] = np.where(df["ball_still_long"].to_numpy(), W_STILL, 0)
    comp["isolated"] = np.where(df["ball_isolated"].to_numpy(), W_ISOLATED, 0)
    gather = df["crowd"].to_numpy() | df["dense_cluster"].to_numpy() | df["cooling_break"].to_numpy()
    comp["gather"] = np.where(gather, W_GATHER, 0)
    comp["blackout"] = np.where(df["track_blackout"].to_numpy(), W_BLACKOUT, 0)
    comp["surplus"] = np.where(df["h_surplus"].to_numpy() | df["a_surplus"].to_numpy(), W_SURPLUS, 0)
    lost = _lost_near_line(df, x, y, obs, vx, vy)
    comp["lost_near_line"] = np.where(lost, W_LOST_LINE, 0)
    goals, goal_mask = _goals(df, tg, x, y, obs)
    comp["goal"] = np.where(goal_mask, W_GOAL, 0)
    restart = restart_picture(df, ev, tg, speed, comp["ball_moving"] != 0, comp["contact"] != 0)
    hold = np.zeros(len(df), dtype=bool)
    for e in ev.loc[ev["kind"] == "exit", "frame"].to_numpy():
        hold[e:_hold(len(df), e, restart, OUT_HOLD_MAX)] = True
    comp["out_hold"] = np.where(hold & df["graded"].to_numpy(), W_OUT_HOLD, 0)
    other_dead = df["ball_still_long"].to_numpy() | gather | lost
    stop, onsets = _stoppages_from_slowdown(df, speed, comp["ball_moving"] != 0, other_dead, restart)
    comp["stop_hold"] = np.where(stop, W_STOP_HOLD, 0)

    score = np.zeros(len(df))
    for v in comp.values():
        score = score + v
    return score, comp, {"goals": goals, "lost_near_line": lost, "slowdown_onsets": onsets}


def decode(score, graded, forced_dead):
    """Two-state Viterbi per graded block: maximise sum of score over dead frames - SWITCH_COST * switches."""
    n = len(score)
    dead = np.zeros(n, dtype=bool)
    st, en, val = runs(graded.astype(np.int8))
    for a, b, v in zip(st, en, val):
        if not v:
            continue
        m = b - a
        s = score[a:b]
        fd = forced_dead[a:b]
        back = np.zeros((m, 2), dtype=np.int8)       # previous state for (live, dead)
        live, dd = 0.0, s[0]
        if fd[0]:
            live = -np.inf
        for i in range(1, m):
            l_from_l, l_from_d = live, dd - SWITCH_COST
            d_from_d, d_from_l = dd, live - SWITCH_COST
            if l_from_l >= l_from_d:
                nl, back[i, 0] = l_from_l, 0
            else:
                nl, back[i, 0] = l_from_d, 1
            if d_from_d >= d_from_l:
                nd, back[i, 1] = d_from_d + s[i], 1
            else:
                nd, back[i, 1] = d_from_l + s[i], 0
            live, dd = (-np.inf if fd[i] else nl), nd
        state = 1 if dd >= live else 0
        path = np.zeros(m, dtype=bool)
        for i in range(m - 1, -1, -1):
            path[i] = state == 1
            state = back[i, state] if i > 0 else state
        dead[a:b] = path
    return dead


def _segments(dead, graded, df, ev, comp, extra, tg):
    """One row per dead stretch: when, how long, what started it, what ended it."""
    rows = []
    exits = ev.loc[ev["kind"] == "exit", "frame"].to_numpy()
    entries = ev.loc[ev["kind"] == "entry", "frame"].to_numpy()
    kick = tg[(tg["code_key"] == "set pieces") & tg["type"].str.lower().str.startswith("kick off")]
    kicks = kick["start_frame"].to_numpy()
    goal_starts = np.array([g for g, _ in extra["goals"]])
    st, en = runs_true(dead & graded)
    names = ["goal", "out_hold", "stop_hold", "ball_out", "lost_near_line", "gather", "still", "isolated", "blackout"]
    for a, b in zip(st, en):
        w = slice(max(0, a - FPS), min(len(dead), a + 2 * FPS))
        if df["known_dead"].to_numpy()[a]:
            onset = "before 2nd-half kickoff"
        elif len(goal_starts) and np.any(np.abs(goal_starts - a) <= 3 * FPS):
            onset = "goal"
        elif np.any((exits >= a - FPS) & (exits <= a + 2 * FPS)):
            onset = "ball over a line"
        elif comp["lost_near_line"][w].any():
            onset = "ball lost near a line"
        elif comp["ball_out"][w].any():
            onset = "ball seen outside"
        elif comp["gather"][a:b].any():
            onset = "crowd / cooling break"
        elif comp["stop_hold"][w].any():
            onset = "sudden slowdown (foul?)"
        else:
            onset = "players slow down"
        if np.any(np.abs(kicks - b) <= 5 * FPS):
            restart = "kickoff"
        elif np.any((entries >= b - 3 * FPS) & (entries <= b + FPS)):
            restart = "ball re-enters (throw-in?)"
        elif df["graded"].to_numpy()[min(b, len(dead) - 1)] == 0 or b == len(dead):
            restart = "end of half"
        else:
            restart = "play resumes"
        cues = [k for k in names if np.any(comp[k][a:b] != 0)]
        rows.append({"start": int(a), "end": int(b), "seconds": (b - a) / FPS, "onset": onset,
                     "restart": restart, "cues": ",".join(cues)})
    return pd.DataFrame(rows)


def report(seg, dead, graded, score, extra, path):
    L_ = []
    w = L_.append
    g = graded
    gm = g.sum() / FPS / 60
    dm = (dead & g).sum() / FPS / 60
    w("Dead-ball detector -- report (aggregates and timestamps only, safe to share)")
    w("=" * 88)
    w("graded time {:.1f} min | dead {:.1f} min ({:.1%}) | live (ball in play) {:.1f} min".format(
        gm, dm, dm / gm, gm - dm))
    w("stoppages: {}  | length p10/p50/p90 {:.1f} / {:.1f} / {:.1f} s | under 5 s: {}, 5-15 s: {}, 15-60 s: {}, over 60 s: {}".format(
        len(seg), *np.percentile(seg["seconds"], [10, 50, 90]),
        int((seg.seconds < 5).sum()), int(((seg.seconds >= 5) & (seg.seconds < 15)).sum()),
        int(((seg.seconds >= 15) & (seg.seconds < 60)).sum()), int((seg.seconds >= 60).sum())))
    w("for reference, a professional match typically has ~52-58 min of ball in play and ~100-130 stoppages")
    w("goals the data agrees with: {} ({})".format(
        len(extra["goals"]), ", ".join(video_time(a) for a, _ in extra["goals"])))
    w("")
    w("how stoppages start:    " + "  ".join("{} {}".format(k, v) for k, v in seg["onset"].value_counts().items()))
    w("how stoppages end:      " + "  ".join("{} {}".format(k, v) for k, v in seg["restart"].value_counts().items()))
    w("")
    w("TIMESTAMPS TO CHECK (debugging only: does the code do what the rules say?)")

    def line(r):
        return "  {} -> {} | clock {} | {:5.1f} s | starts: {:<24} | ends: {:<26} | cues: {}".format(
            video_time(r.start), video_time(r.end), match_clock(r.start), r.seconds, r.onset, r.restart,
            r.cues or "-")
    w(" shortest:")
    for r in seg.nsmallest(5, "seconds").sort_values("start").itertuples():
        w(line(r))
    w(" longest:")
    for r in seg.nlargest(5, "seconds").sort_values("start").itertuples():
        w(line(r))
    w(" spread over the match:")
    for r in seg.iloc[np.linspace(0, len(seg) - 1, 10).astype(int)].itertuples():
        w(line(r))
    w(" live stretches with the most dead-ball evidence (closest to being called dead):")
    st, en = runs_true(~dead & g)
    near = sorted(((score[a:b].clip(0).sum(), a, b) for a, b in zip(st, en) if b - a > FPS), reverse=True)[:5]
    for s, a, b in sorted(near, key=lambda t: t[1]):
        k = a + int(np.argmax(pd.Series(score[a:b]).rolling(3 * FPS, min_periods=1).sum().to_numpy()))
        w("  around {} (clock {}) inside a live stretch {} -> {}".format(
            video_time(k), match_clock(k), video_time(a), video_time(b)))
    text = "\n".join(L_) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=EC.CACHE_DIR)
    args = ap.parse_args(argv)
    df = pd.read_parquet(os.path.join(args.cache, "frames.parquet"))
    ev = pd.read_parquet(os.path.join(args.cache, "events.parquet"))
    tg = pd.read_parquet(os.path.join(args.cache, "tagging.parquet"))
    graded = df["graded"].to_numpy()

    speed = measured_speed(args.cache)
    score, comp, extra = evidence(df, ev, tg, speed)
    dead = decode(score, graded, df["known_dead"].to_numpy())
    seg = _segments(dead, graded, df, ev, comp, extra, tg)

    out = pd.DataFrame({"frame": df["frame"].to_numpy(), "dead_score": score.astype(np.float32),
                        "dead": np.where(graded, dead.astype(np.int8), -1).astype(np.int8)})
    out.to_parquet(os.path.join(args.cache, "deadball.parquet"), index=False)
    seg.to_parquet(os.path.join(args.cache, "deadball_segments.parquet"), index=False)
    print(report(seg, dead, graded, score, extra, os.path.join(args.cache, "deadball_report.txt")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
