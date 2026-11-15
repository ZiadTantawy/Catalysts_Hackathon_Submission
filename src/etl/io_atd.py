"""Stage 1: the ATD pair (metadata XML + raw text) into numpy arrays, in one streaming pass.

Raw line format (see starter/baseline.py):  "<frame_id>:<slot>;<slot>;...;<slot>:<ball>"
where each slot is "x,y,speed" (normalised 0-1 coordinates, m/s) and missing values are "NaN".
frame_id counts from 1; submission row = frame_id - 1.

The parser does not assume the channel layout. It reads it from the XML's
DataFormatSpecification when one is present, falls back to the order of <Player> elements (what the
official baseline does), and reports whether the two agree. A line whose field count does not match
is parsed group by group instead of failing, and counted.
"""

import re
import xml.etree.ElementTree as ET

import numpy as np

from . import config as C

HOME, AWAY = 0, 1
# Words that would reveal a ball-state or ball-owner channel in the export. If any appear, the
# report says so -- that would change the whole approach.
SUSPECT_WORDS = re.compile(r"owning|owner|possess|status|state|alive|dead|inplay|in_play", re.I)


def _strip_ns(root):
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def read_meta(path):
    """Slot order, team of each slot, channel names and anything that looks like a state channel."""
    root = _strip_ns(ET.parse(path).getroot())

    teams = [(t.get("id"), (t.findtext("Name") or "").strip()) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(path, len(teams)))
    home_id = teams[0][0]                                   # EPTS / kloppy convention

    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    team_of_player = dict(players)
    player_order = [pid for pid, _ in players]

    # channel order from the data specification, if the export has one
    channel_player, channel_name = {}, {}
    for pc in root.iter("PlayerChannel"):
        channel_player[pc.get("id")] = pc.get("playerId")
        channel_name[pc.get("id")] = pc.get("channelId")
    spec_order, spec_channels = [], {}
    for ref in root.iter("PlayerChannelRef"):
        cid = ref.get("playerChannelId")
        pid = channel_player.get(cid)
        if pid is None:
            continue
        if pid not in spec_channels:
            spec_order.append(pid)
            spec_channels[pid] = []
        spec_channels[pid].append(channel_name.get(cid, cid))
    ball_channels = [ref.get("channelId") for ref in root.iter("BallChannelRef")]

    if spec_order:
        slot_players = spec_order
        player_channels = spec_channels[spec_order[0]]
    else:
        slot_players = player_order
        player_channels = ["x", "y", "speed"]

    slot_team = np.array([HOME if team_of_player.get(pid) == home_id else AWAY
                          for pid in slot_players], dtype=np.int8)

    gk_players = set()
    for p in root.iter("Player"):
        text = " ".join(t for t in p.itertext()).lower()
        if "goalkeeper" in text or "goal keeper" in text:
            gk_players.add(p.get("id"))
    slot_is_gk = np.array([pid in gk_players for pid in slot_players], dtype=bool)

    suspects = sorted({m.group(0).lower()
                       for el in root.iter()
                       for text in [el.tag, *(el.attrib.values()), el.text or ""]
                       if isinstance(text, str)
                       for m in SUSPECT_WORDS.finditer(text)})

    return {
        "teams": teams,
        "home_name": teams[0][1],
        "away_name": teams[1][1],
        "slot_players": slot_players,
        "slot_team": slot_team,
        "slot_is_gk": slot_is_gk,
        "player_channels": player_channels,
        "ball_channels": ball_channels or ["x", "y", "speed"],
        "slot_order_source": "DataFormatSpecification" if spec_order else "Player elements",
        "slot_order_agrees_with_player_list": (not spec_order) or spec_order == player_order,
        "suspect_words": suspects,
    }


def _parse_group_list(text, n_groups, width):
    """Slow, tolerant path: one group at a time; empty or short groups become NaN."""
    out = np.full((n_groups, width), np.nan, dtype=np.float32)
    for i, grp in enumerate(text.split(";")[:n_groups]):
        vals = grp.split(",")
        for j, v in enumerate(vals[:width]):
            v = v.strip()
            if v:
                try:
                    out[i, j] = float(v)
                except ValueError:
                    pass
    return out


def read_raw(path, n_slots, n_player_ch, n_ball_ch, n_frames=C.N_FRAMES):
    """(players[T, S, Cp], ball[T, Cb], info). Row t is video frame t (= frame_id - 1)."""
    players = np.full((n_frames, n_slots, n_player_ch), np.nan, dtype=np.float32)
    ball = np.full((n_frames, n_ball_ch), np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)
    info = {"n_lines": 0, "n_slow_lines": 0, "n_bad_lines": 0, "n_out_of_range": 0,
            "n_duplicate_frames": 0, "first_frame_id": None, "last_frame_id": None}
    p_width = n_slots * n_player_ch

    def flush(ids, pl_txt, bl_txt):
        ids = np.asarray(ids, dtype=np.int64) - 1
        ok = (ids >= 0) & (ids < n_frames)
        info["n_out_of_range"] += int((~ok).sum())
        info["n_duplicate_frames"] += int(seen[ids[ok]].sum())
        flat = ",".join(pl_txt).replace(";", ",").split(",")
        if len(flat) == len(ids) * p_width:
            arr = np.array(flat, dtype=np.float32).reshape(len(ids), n_slots, n_player_ch)
        else:                                            # ragged chunk: line by line
            arr = np.empty((len(ids), n_slots, n_player_ch), dtype=np.float32)
            for k, txt in enumerate(pl_txt):
                vals = txt.replace(";", ",").split(",")
                if len(vals) == p_width:
                    arr[k] = np.array(vals, dtype=np.float32).reshape(n_slots, n_player_ch)
                else:
                    info["n_slow_lines"] += 1
                    arr[k] = _parse_group_list(txt, n_slots, n_player_ch)
        bflat = ",".join(bl_txt).split(",")
        if len(bflat) == len(ids) * n_ball_ch:
            barr = np.array(bflat, dtype=np.float32).reshape(len(ids), n_ball_ch)
        else:
            barr = np.vstack([_parse_group_list(b, 1, n_ball_ch) for b in bl_txt])
        players[ids[ok]] = arr[ok]
        ball[ids[ok]] = barr[ok]
        seen[ids[ok]] = True

    ids, pl, bl = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            info["n_lines"] += 1
            if len(parts) != 3:
                info["n_bad_lines"] += 1
                continue
            fid = int(parts[0])
            if info["first_frame_id"] is None:
                info["first_frame_id"] = fid
            info["last_frame_id"] = fid
            ids.append(fid)
            pl.append(parts[1])
            bl.append(parts[2])
            if len(ids) >= C.PARSE_CHUNK:
                flush(ids, pl, bl)
                ids, pl, bl = [], [], []
    if ids:
        flush(ids, pl, bl)

    info["n_frames_covered"] = int(seen.sum())
    info["contiguous_from_1"] = bool(
        info["first_frame_id"] == 1 and info["last_frame_id"] == n_frames
        and info["n_frames_covered"] == n_frames and info["n_duplicate_frames"] == 0)
    return players, ball, info


def sniff_widths(path):
    """(n_slots, n_player_channels, n_ball_channels) from the first data line."""
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(":")
            if len(parts) == 3:
                groups = parts[1].split(";")
                return len(groups), len(groups[0].split(",")), len(parts[2].split(","))
    raise ValueError("{}: no parsable line".format(path))
