"""DFL event log -> control events on the positions timeline, and the turnover check for labels.

The event log is human-made (DFL), independent of the tracking flag `BallPossession`, so it is the
outside reference for judging a possession-label rule. A *control event* is one where a team clearly
had the ball: a pass / cross attempt (`Play`, any outcome, also inside a restart), a shot, an
interception or claimed ball (`BallClaiming`), or a tackle won with the ball (`TacklingGame` winner
`withBallControl`). `OtherBallAction` (an unspecified touch) is not counted: a touch is not control.

    ev = events.read_events(path, info)              # one row per event
    ctl = events.control_events(ev, m)               # control events with a frame row, synced
    tov = events.event_turnovers(ctl, codes)         # team A's last action -> team B's first action
    events.turnover_check(codes, half, tov)          # do the label's changes match them?

Event times are wall-clock times entered live, so they are offset from the tracking by up to a few
seconds. `estimate_offset` finds, per match, the shift that puts the most passers on the ball.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import labels, parse

FPS = parse.FPS
RESTARTS = {"KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty"}


def read_events(path, info):
    team_idx = {tid: t["idx"] for tid, t in info["teams"].items()}
    rows = []
    for ev in ET.parse(path).getroot().iter("Event"):
        kids = list(ev)
        if not kids:
            continue
        top = kids[0]
        play = top if top.tag == "Play" else top.find(".//Play")
        shot = top if top.tag == "ShotAtGoal" else top.find(".//ShotAtGoal")
        row = {"time": ev.get("EventTime"), "kind": top.tag, "team": -1, "player": None,
               "evaluation": None, "recipient": None, "control": False}
        if play is not None:
            row.update(team=team_idx.get(play.get("Team"), -1), player=play.get("Player"),
                       evaluation=play.get("Evaluation"), recipient=play.get("Recipient"),
                       control=True, action="pass")
        elif shot is not None:
            row.update(team=team_idx.get(shot.get("Team"), -1), player=shot.get("Player"),
                       control=True, action="shot")
        elif top.tag == "BallClaiming":
            row.update(team=team_idx.get(top.get("Team"), -1), player=top.get("Player"),
                       control=True, action="claim")
        elif top.tag == "TacklingGame":
            won = top.get("WinnerRole") == "withBallControl"
            row.update(team=team_idx.get(top.get("WinnerTeam"), -1), player=top.get("Winner"),
                       control=won, action="tackle_won" if won else "tackle")
        else:
            row.update(team=team_idx.get(top.get("Team"), -1), player=top.get("Player"),
                       action=top.tag)
        row["restart"] = top.tag in RESTARTS
        rows.append(row)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def event_rows(m, times, offset_s=0.0):
    """Frame row of each event time (NaN outside the tracked sections)."""
    out = np.full(len(times), np.nan)
    t = pd.to_datetime(pd.Series(times), utc=True).dt.tz_convert(None).to_numpy()
    for k, sec in enumerate(m["meta"]["sections"], start=1):
        row0, t0 = m["meta"]["ball_t0"][sec]
        rows = np.flatnonzero(m["half"] == k)
        r = row0 + np.round(((t - np.datetime64(pd.Timestamp(t0).tz_convert(None)))
                             / np.timedelta64(1, "ms") / 1000 + offset_s) * FPS)
        ok = (r >= rows[0]) & (r <= rows[-1])
        out[ok] = r[ok]
    return out


def _passer_on_ball(m, ev, offset_s, max_dist=1.5):
    p = ev[(ev.action == "pass") & ev.player.notna()]
    rows = event_rows(m, p.time.dt.tz_convert(None), offset_s)
    pidx = {pid: j for j, pid in enumerate(m["person_id"])}
    hit = n = 0
    for r, pl in zip(rows, p.player):
        if np.isnan(r) or pl not in pidx:
            continue
        r = int(r)
        d = np.hypot(*(m["xy"][r, pidx[pl]] - m["ball"][r, :2]))
        n += 1
        hit += d <= max_dist
    return hit / max(n, 1)


def estimate_offset(m, ev, grid=np.arange(-3.0, 3.01, 0.2)):
    """Shift (s) added to event times that maximises the share of passes with the passer on the ball."""
    scores = [_passer_on_ball(m, ev, o) for o in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best]), float(scores[int(np.argmin(np.abs(grid)))])


def snap_to_player(m, rows, players, window_s=2.0, max_dist=2.0):
    """Move each event to the frame within +-window_s where its player is closest to the ball, if they
    get within max_dist m; otherwise keep the row. Returns new rows and whether each was snapped."""
    pidx = {pid: j for j, pid in enumerate(m["person_id"])}
    w = int(window_s * FPS)
    bounds = {h: (np.flatnonzero(m["half"] == h)[0], np.flatnonzero(m["half"] == h)[-1] + 1)
              for h in np.unique(m["half"])}
    out, snapped = rows.copy(), np.zeros(len(rows), bool)
    for k, (r, pl) in enumerate(zip(rows, players)):
        if pl not in pidx:
            continue
        h0, h1 = bounds[m["half"][r]]  # stay inside the half
        lo, hi = max(r - w, h0), min(r + w + 1, h1)
        d = np.hypot(*(m["xy"][lo:hi, pidx[pl]] - m["ball"][lo:hi, :2]).T)
        if np.isfinite(d).any() and np.nanmin(d) <= max_dist:
            out[k], snapped[k] = lo + int(np.nanargmin(d)), True
    return out, snapped


def control_events(ev, m, offset_s=0.0, snap=True):
    c = ev[ev.control & (ev.team >= 0)].copy()
    c["row"] = event_rows(m, c.time.dt.tz_convert(None), offset_s)
    c = c[c.row.notna()].copy()
    c["row"] = c.row.astype(np.int64)
    c["snapped"] = False
    if snap:
        c["row"], c["snapped"] = snap_to_player(m, c.row.to_numpy(), c.player.to_numpy())
    return c.sort_values("row", kind="stable").reset_index(drop=True)


def event_turnovers(ctl, codes):
    """Consecutive control events of different teams with live ball all the way between them.
    Returns (row of A's last action, row of B's first action, B's team)."""
    out = []
    r, t = ctl.row.to_numpy(), ctl.team.to_numpy()
    dead_cum = np.r_[0, np.cumsum(codes == labels.D)]
    for i in range(1, len(r)):
        if t[i] == t[i - 1] or r[i] <= r[i - 1]:
            continue
        if dead_cum[r[i] + 1] - dead_cum[r[i - 1]] == 0:  # no dead frame from A's action to B's
            out.append((r[i - 1], r[i], t[i]))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def label_turnovers(codes, half):
    """Rows where the label changes directly between H and A inside a half: (row, new team)."""
    c = codes
    i = np.flatnonzero((c[1:] != c[:-1]) & (c[1:] != labels.D) & (c[:-1] != labels.D)
                       & (half[1:] == half[:-1])) + 1
    return np.stack([i, c[i]], axis=1)


def _confirmed(lt, tov, tol):
    confirmed = np.zeros(len(lt), bool)
    if len(lt) and len(tov):
        for a, b, team in tov:
            lo, hi = np.searchsorted(lt[:, 0], [a - tol, b + tol + 1])
            confirmed[np.flatnonzero(lt[lo:hi, 1] == team) + lo] = True
    return confirmed


def event_confirmed_relabel(codes, half, tov, tol=int(0.5 * FPS)):
    """Variant B: inside each stretch of live play, possession changes only at label turnovers that an
    event turnover confirms (the new team made a control action, see turnover_check). Every other
    change is ignored, so the team before it keeps the ball. Dead frames and the team at each restart
    are left as the tracking flag says."""
    lt = label_turnovers(codes, half)
    keep = _confirmed(lt, tov, tol)
    switch_at = dict(zip(lt[keep, 0], lt[keep, 1]))
    out = codes.copy()
    live = codes != labels.D
    start = np.flatnonzero(live & np.r_[True, ~live[:-1] | (half[1:] != half[:-1])])
    for s in start:
        e = s
        team = codes[s]
        while e < len(codes) and live[e] and half[e] == half[s]:
            team = switch_at.get(e, team)
            out[e] = team
            e += 1
    return out


def turnover_check(codes, half, tov, tol=int(0.5 * FPS)):
    """Do the label's live turnovers match the event log's?

    An event turnover is found when the label switches to team B somewhere between A's last action and
    B's first action (+- tol frames). A label turnover is confirmed when it lies inside such a window
    with the matching new team. Returns recall, precision, and the counts."""
    lt = label_turnovers(codes, half)
    found = np.zeros(len(tov), bool)
    confirmed = np.zeros(len(lt), bool)
    if len(lt) and len(tov):
        for k, (a, b, team) in enumerate(tov):
            lo, hi = np.searchsorted(lt[:, 0], [a - tol, b + tol + 1])
            hit = np.flatnonzero(lt[lo:hi, 1] == team) + lo
            if len(hit):
                found[k] = True
                confirmed[hit] = True
    return {"event_turnovers": len(tov), "label_turnovers": len(lt),
            "recall": found.mean() if len(tov) else np.nan,
            "precision": confirmed.mean() if len(lt) else np.nan}
