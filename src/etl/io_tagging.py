"""Stage 9: Smart Tagging intervals in video frames, and their per-frame projection.

Time base (verified by the organisers, re-checked in checks.py): <start>/<end> are seconds from the
first frame of the tactical video, so frame = round(seconds * 25). Intervals are half-open
[start_frame, end_frame).

Team values are club names; they are mapped to sides by matching the ATD metadata's team names
(first team = home), with a spelling-tolerant fallback, and "N/A" kept as its own value. Every code
and tag value that could not be mapped is reported rather than dropped.
"""

import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import config as C


def slug(code):
    return re.sub(r"[^a-z0-9]+", "_", code.strip().lower()).strip("_")


def _side_of(value, home_name, away_name):
    v = (value or "").strip().lower()
    if not v:
        return "none"
    if v in ("n/a", "na", "none"):
        return "N/A"
    if v == home_name.lower() or "barcelona" in v or "barça" in v:
        return "H"
    if v == away_name.lower() or "athletic" in v or "atletic" in v or "bilbao" in v:
        return "A"
    return "unmapped"


def load(path, home_name, away_name):
    rows = []
    for inst in ET.parse(path).iter("instance"):
        start_s = float(inst.findtext("start"))
        end_s = float(inst.findtext("end"))
        code = (inst.findtext("code") or "").strip()
        tags = {}
        for lab in inst.findall("label"):
            tags[(lab.findtext("group") or "").strip()] = (lab.findtext("text") or "").strip()
        tag_ci = {k.lower(): v for k, v in tags.items()}
        rows.append({
            "code": code,
            "code_key": code.lower(),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * C.FPS)),
            "end_frame": int(round(end_s * C.FPS)),
            "team_raw": tag_ci.get("team", ""),
            "side": _side_of(tag_ci.get("team", ""), home_name, away_name),
            "half_tag": tag_ci.get("half", ""),
            "type": tag_ci.get("type", ""),
            "tags_json": json.dumps(tags, sort_keys=True, ensure_ascii=False),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["start_frame", "end_frame", "code"], kind="mergesort").reset_index(drop=True)
    return df


def per_frame(iv, n=C.N_FRAMES):
    """Per-frame tagging columns with a fixed schema (known codes first, unknown ones appended)."""
    known = list(C.TAG_POSSESSION_SIGN)
    extra = sorted(set(iv["code_key"]) - set(known))
    cols = {}
    vote = np.zeros(n, dtype=np.int16)
    side_val = {"H": C.TEAM_SIGN_HOME, "A": C.TEAM_SIGN_AWAY}
    for code in known + extra:
        active = np.zeros(n, dtype=np.int8)
        team = np.zeros(n, dtype=np.int8)
        sub = iv[iv["code_key"] == code]
        for r in sub.itertuples():                        # sorted by start: later overwrite
            s, e = max(r.start_frame, 0), min(r.end_frame, n)
            if e <= s:
                continue
            active[s:e] = 1
            team[s:e] = side_val.get(r.side, C.TEAM_SIGN_NONE)
        key = slug(code)
        cols["tag_" + key] = active
        cols["tag_" + key + "_team"] = team
        vote += C.TAG_POSSESSION_SIGN.get(code, 0) * team
    cols["tag_vote_home"] = np.clip(vote, -127, 127).astype(np.int8)

    # set pieces: active type, and distance in frames to the previous / next set-piece start
    sp = iv[iv["code_key"] == C.SET_PIECE_CODE]
    type_idx = {t: i for i, t in enumerate(C.SET_PIECE_TYPES)}
    sp_type = np.full(n, -1, dtype=np.int8)
    starts = np.zeros(n, dtype=bool)
    start_type = np.full(n, -1, dtype=np.int8)
    for r in sp.itertuples():
        s, e = max(r.start_frame, 0), min(r.end_frame, n)
        ti = type_idx.get(r.type.strip().lower(), len(C.SET_PIECE_TYPES))
        if e > s:
            sp_type[s:e] = ti
        if 0 <= r.start_frame < n:
            starts[r.start_frame] = True
            start_type[r.start_frame] = ti
    cols["setpiece_type_active"] = sp_type
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(starts, idx, -1))
    nxt = np.minimum.accumulate(np.where(starts, idx, n)[::-1])[::-1]
    cols["frames_since_setpiece_start"] = np.where(prev >= 0, idx - prev, -1).astype(np.int32)
    cols["frames_until_setpiece_start"] = np.where(nxt < n, nxt - idx, -1).astype(np.int32)
    nxt_c = np.clip(nxt, 0, n - 1)
    cols["next_setpiece_type"] = np.where(nxt < n, start_type[nxt_c], -1).astype(np.int8)

    goals = np.zeros(n, dtype=bool)
    for r in iv[iv["code_key"] == C.GOALS_CODE].itertuples():
        if 0 <= r.start_frame < n:
            goals[r.start_frame] = True
    gprev = np.maximum.accumulate(np.where(goals, idx, -1))
    cols["frames_since_goal_tag"] = np.where(gprev >= 0, idx - gprev, -1).astype(np.int32)
    return cols, extra
