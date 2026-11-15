"""IDSSE (DFL Bundesliga) positions and match information -> per-match arrays.

    from src.idsse import parse
    parse.inventory()                  # which matches are complete on disk
    m = parse.load_match("J03WOH")     # parses once (~10-20 s), then reads cache/idsse/J03WOH.npz

Data: Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer",
Scientific Data (2025). Files are found under $IDSSE_DIR, else the first `external-data/idsse`
folder above this file. Raw files and the cache are never committed (.gitignore).

Positions file layout: one <FrameSet> per person per game section, 25 Hz. Frame attributes, in
order: N (frame counter), T, X, Y, [Z: ball only], D, S (speed, km/h), A, M (match minute), and on the
ball only BallPossession and BallStatus. Coordinates are metres with the origin at the centre spot.

Arrays of a parsed match (T frames, P persons, both sections concatenated in order):
    half    int8  [T]        1, 2 (3, 4 for extra time if present)
    n_dfl   int32 [T]        DFL frame counter N
    minute  int16 [T]        match minute M of the ball frame (-1 where the ball is missing)
    xy      f32   [T, P, 2]  player positions, NaN when the person is not on the pitch
    speed   f32   [T, P]     player speed (m/s, converted from DFL's km/h)
    ball    f32   [T, 4]     x, y, z, speed (m/s)
    poss    int8  [T]        raw BallPossession (1 / 2, 0 where missing)
    status  int8  [T]        raw BallStatus (1 alive / 0 dead, -1 where missing)
    person_id    str  [P]
    person_team  int8 [P]    0 home · 1 away (the TEAM_IDX convention of the ETL)
    person_gk    bool [P]    PlayingPosition == "TW" in the match information
"""

import glob
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(REPO_ROOT, "cache", "idsse")
CACHE_VERSION = 3
KMH_PER_MS = 3.6   # DFL's S is km/h (3.6x the speed from positions); stored as m/s like the ETL

_FRAMESET = re.compile(rb'<FrameSet ([^>]*)>')
_ATTR = re.compile(rb'(\w+)="([^"]*)"')
_FRAME = re.compile(
    rb'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)" (?:Z="([-\d.]+)" )?'
    rb'D="[^"]*" S="([-\d.]+)" A="[^"]*" M="(-?\d+)"'
    rb'(?: BallPossession="(\d+)" BallStatus="(\d+)")?')
_SECTION_ORDER = ["firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra"]


# ---- files ---------------------------------------------------------------------------------------

def find_data_dir():
    env = os.environ.get("IDSSE_DIR")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(d, "external-data", "idsse")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError("IDSSE data not found: set IDSSE_DIR")
        d = parent


def _match_id(path):
    return os.path.splitext(os.path.basename(path))[0].split("DFL-MAT-")[-1]


def _tail_has(path, token, n=512):
    with open(path, "rb") as fh:
        fh.seek(max(0, os.path.getsize(path) - n))
        return token in fh.read()


def files(data_dir=None):
    """match id -> {"info", "events", "positions"} paths (None where the file is absent)."""
    data_dir = data_dir or find_data_dir()
    kinds = {"info": "DFL_02_01_matchinformation_*", "events": "DFL_03_02_events_raw_*",
             "positions": "DFL_04_03_positions_raw_observed_*"}
    out = {}
    for kind, pat in kinds.items():
        for p in glob.glob(os.path.join(data_dir, pat + ".xml")):
            out.setdefault(_match_id(p), dict.fromkeys(kinds))[kind] = p
    return dict(sorted(out.items()))


def inventory(data_dir=None):
    """One row per match: which files exist and whether each is complete (closing tag present)."""
    rows = []
    for mid, f in files(data_dir).items():
        row = {"match": mid}
        for kind in ("info", "events", "positions"):
            p = f[kind]
            row[kind + "_mb"] = round(os.path.getsize(p) / 1e6, 1) if p else None
            row[kind + "_ok"] = bool(p) and _tail_has(p, b"</PutDataRequest>")
        row["usable"] = row["info_ok"] and row["positions_ok"]
        rows.append(row)
    return rows


# ---- match information ---------------------------------------------------------------------------

def read_matchinfo(path):
    root = ET.parse(path).getroot()
    general = root.find(".//General").attrib
    info = {"match_id": general.get("MatchId", _match_id(path)),
            "competition": general.get("CompetitionName"), "season": general.get("Season"),
            "matchday": general.get("MatchDay"), "result": general.get("Result"),
            "teams": {}, "persons": {}}
    for team in root.iter("Team"):
        role = team.get("Role")
        idx = 0 if role == "home" else 1
        info["teams"][team.get("TeamId")] = {"name": team.get("TeamName"), "role": role, "idx": idx}
        for pl in team.iter("Player"):
            info["persons"][pl.get("PersonId")] = {
                "team": idx, "shirt": pl.get("ShirtNumber"), "name": pl.get("Shortname"),
                "position": pl.get("PlayingPosition"), "starting": pl.get("Starting") == "true"}
    info["home_name"] = next(t["name"] for t in info["teams"].values() if t["idx"] == 0)
    info["away_name"] = next(t["name"] for t in info["teams"].values() if t["idx"] == 1)
    return info


# ---- positions -----------------------------------------------------------------------------------

def _parse_block(block, is_ball):
    rows = _FRAME.findall(block)
    if len(rows) != block.count(b"<Frame "):
        raise ValueError("frame regex matched {} of {} frames".format(len(rows), block.count(b"<Frame ")))
    a = np.array(rows, dtype="S16")
    out = {"n": a[:, 0].astype(np.int64), "x": a[:, 1].astype(np.float32),
           "y": a[:, 2].astype(np.float32), "s": a[:, 4].astype(np.float32) / KMH_PER_MS,
           "m": a[:, 5].astype(np.int16)}
    if is_ball:
        out["z"] = a[:, 3].astype(np.float32)
        out["poss"] = a[:, 6].astype(np.int8)
        out["status"] = a[:, 7].astype(np.int8)
    return out


def parse_positions(path, info):
    """Parse one positions XML into the arrays described in the module docstring."""
    with open(path, "rb") as fh:
        data = fh.read()
    if b"</PutDataRequest>" not in data[-512:]:
        raise ValueError("positions file is truncated: " + path)
    pitch = re.search(rb'<PitchSize X="([\d.]+)" Y="([\d.]+)"', data)
    heads = list(_FRAMESET.finditer(data))
    sets, ball_t0 = [], {}
    for i, h in enumerate(heads):
        attrs = {k.decode(): v.decode() for k, v in _ATTR.findall(h.group(1))}
        end = heads[i + 1].start() if i + 1 < len(heads) else len(data)
        if attrs["TeamId"] == "referee":
            continue
        is_ball = attrs["TeamId"] == "BALL"
        block = data[h.end():end]
        if is_ball:  # wall-clock time of the first ball frame, to place events on the timeline
            first = re.search(rb'<Frame N="(\d+)" T="([^"]+)"', block)
            ball_t0[attrs["GameSection"]] = (int(first.group(1)), first.group(2).decode())
        sets.append((attrs, _parse_block(block, is_ball)))
    del data, block

    sections = sorted({a["GameSection"] for a, _ in sets},
                      key=lambda s: _SECTION_ORDER.index(s) if s in _SECTION_ORDER else 99)
    # timeline: per section, every DFL frame between the first and last seen by anyone
    offset, sec_n0, halves, n_dfl = 0, {}, [], []
    for k, sec in enumerate(sections, start=1):
        ns = [b["n"] for a, b in sets if a["GameSection"] == sec]
        n0, n1 = min(int(n.min()) for n in ns), max(int(n.max()) for n in ns)
        sec_n0[sec] = (offset, n0)
        halves.append(np.full(n1 - n0 + 1, k, np.int8))
        n_dfl.append(np.arange(n0, n1 + 1, dtype=np.int32))
        offset += n1 - n0 + 1
    T = offset

    persons = sorted({a["PersonId"] for a, _ in sets if a["TeamId"] != "BALL"},
                     key=lambda p: (info["persons"].get(p, {}).get("team", 9), p))
    pidx = {p: i for i, p in enumerate(persons)}
    team_of_set = {}
    for a, _ in sets:
        if a["TeamId"] != "BALL":
            team_of_set[a["PersonId"]] = info["teams"][a["TeamId"]]["idx"]

    xy = np.full((T, len(persons), 2), np.nan, np.float32)
    speed = np.full((T, len(persons)), np.nan, np.float32)
    ball = np.full((T, 4), np.nan, np.float32)
    poss = np.zeros(T, np.int8)
    status = np.full(T, -1, np.int8)
    minute = np.full(T, -1, np.int16)
    for a, b in sets:
        off, n0 = sec_n0[a["GameSection"]]
        rows = off + (b["n"] - n0)
        if a["TeamId"] == "BALL":
            ball[rows] = np.stack([b["x"], b["y"], b["z"], b["s"]], axis=1)
            poss[rows], status[rows], minute[rows] = b["poss"], b["status"], b["m"]
        else:
            j = pidx[a["PersonId"]]
            xy[rows, j, 0], xy[rows, j, 1], speed[rows, j] = b["x"], b["y"], b["s"]

    meta = {k: info[k] for k in ("match_id", "competition", "season", "matchday", "result",
                                 "home_name", "away_name")}
    meta.update(sections=sections, pitch=[float(pitch.group(1)), float(pitch.group(2))] if pitch else None,
                cache_version=CACHE_VERSION,
                # per section: [row of the first ball frame, its wall-clock time T]; frames are 40 ms apart
                ball_t0={s: [sec_n0[s][0] + n - sec_n0[s][1], t] for s, (n, t) in ball_t0.items()})
    return {
        "half": np.concatenate(halves), "n_dfl": np.concatenate(n_dfl), "minute": minute,
        "xy": xy, "speed": speed, "ball": ball, "poss": poss, "status": status,
        "person_id": np.array(persons),
        "person_team": np.array([team_of_set[p] for p in persons], np.int8),
        "person_gk": np.array([info["persons"].get(p, {}).get("position") == "TW" for p in persons]),
        "meta": meta,
    }


def load_match(match_id, data_dir=None, force=False, quiet=False):
    """Parsed arrays for one match, from cache/idsse/<match>.npz when present."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, match_id + ".npz")
    if os.path.exists(cache) and not force:
        z = np.load(cache, allow_pickle=False)
        m = {k: z[k] for k in z.files if k != "meta"}
        m["meta"] = json.loads(str(z["meta"]))
        if m["meta"].get("cache_version") == CACHE_VERSION:
            return m
    f = files(data_dir)[match_id]
    t0 = time.time()
    m = parse_positions(f["positions"], read_matchinfo(f["info"]))
    np.savez(cache, **{k: v for k, v in m.items() if k != "meta"}, meta=json.dumps(m["meta"]))
    if not quiet:
        print("parsed {} in {:.0f} s -> {}".format(match_id, time.time() - t0, cache))
    return m
