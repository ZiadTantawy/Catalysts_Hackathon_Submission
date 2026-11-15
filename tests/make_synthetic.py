"""Generate a synthetic match in the exact on-disk format of the real tactical bundle.

Why this exists: the real match data may not be shared with anyone outside the team (and must not be
uploaded to a hosted LLM), so the ETL is developed and tested against this instead. It
reproduces the *format* faithfully and the *statistics* roughly:

    data/tactical/BAR-ATH_tactical_atd.xml              FIFA/EPTS-style metadata, 46 slots (24 + 22)
    data/tactical/BAR-ATH_tactical_atd.txt              199,319 lines "frame:slot;slot;...:ball"
    data/tactical/BAR-ATH_tactical_smart_tagging.xml    SportsCode-style instances
    data/tactical/BAR-ATH_tactical_homography.csv       a camera that follows the ball

plus tests/out/synthetic_truth.npz, the simulator's own H/A/D state, so feature sanity can be checked
on data whose answer is known. That file is a test fixture only; nothing in src/ reads it.

The simulation is deliberately crude -- 22 dots, a ball that is carried and passed, turnovers,
dead-ball stoppages, a tracker that drops people and swaps their slots, a ball that vanishes ~40%
of the time -- because its job is to exercise every code path, not to be football.

    python tests/make_synthetic.py --out /tmp/synth        # then MTAH_DATA_DIR=/tmp/synth
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = 25
L, W = GAMES[GAME]["pitch"]
N = GAMES[GAME]["n_video_frames"]
N_HOME_SLOTS, N_AWAY_SLOTS = 24, 22
KICKOFF_2ND = 113742

# 4-4-2 in "attacking +x" coordinates, as fractions of the pitch
FORMATION = np.array([
    [0.05, 0.50],
    [0.20, 0.15], [0.18, 0.38], [0.18, 0.62], [0.20, 0.85],
    [0.33, 0.15], [0.30, 0.40], [0.30, 0.60], [0.33, 0.85],
    [0.44, 0.40], [0.44, 0.60],
])


def team_slots():
    return np.arange(N_HOME_SLOTS), np.arange(N_HOME_SLOTS, N_HOME_SLOTS + N_AWAY_SLOTS)


def plan_half(rng, first, last, kickoff, start_team):
    """[(start, end, state)] covering first..last inclusive; state 0=H, 1=A, 2=dead."""
    segs = []
    if kickoff > first:
        segs.append((first, kickoff - 1, 2))
    t, team = kickoff, start_team
    while t <= last:
        dur = int(np.clip(rng.exponential(12.0), 2.0, 60.0) * FPS)
        end = min(last, t + dur - 1)
        segs.append((t, end, team))
        t = end + 1
        if t > last:
            break
        if rng.random() < 0.35:
            dead = int(rng.uniform(6, 40) * FPS)
            end = min(last, t + dead - 1)
            segs.append((t, end, 2))
            t = end + 1
            team = int(rng.integers(0, 2))
        else:
            team = 1 - team
    return segs


def simulate(seed=7):
    rng = np.random.default_rng(seed)
    state = np.full(N, 3, dtype=np.int8)                 # 3 = no match (pre / HT / post)
    blocks = half_video_blocks(GAME)
    kickoffs = {1: blocks[0][1], 2: KICKOFF_2ND}
    segs_all = []
    for half, first, last in blocks:
        segs = plan_half(rng, first, last, kickoffs[half], start_team=half - 1)
        for s, e, st in segs:
            state[s:e + 1] = st
        segs_all.append((half, segs))

    # home attacks +x in half 1 and -x in half 2 (raw coordinates); flip = 180 deg rotation
    def to_raw(xy_att, team, half):
        x, y = xy_att[..., 0], xy_att[..., 1]
        home_att_pos = (half == 1)
        att_pos = home_att_pos if team == 0 else not home_att_pos
        if att_pos:
            return np.stack([x, y], -1)
        return np.stack([L - x, W - y], -1)

    pos = np.full((N, 22, 2), np.nan, dtype=np.float32)    # 0..10 home, 11..21 away
    ball = np.full((N, 2), np.nan, dtype=np.float32)
    height = np.full(N, np.nan, dtype=np.float32)
    form = FORMATION * [L, W]

    cur = np.zeros((22, 2))
    for half, segs in segs_all:
        for k in range(2):
            cur[k * 11:(k + 1) * 11] = to_raw(form, k, half)
        carrier = 10 if segs[0][2] != 1 else 21
        b = np.array([L / 2, W / 2])
        flight = None                                      # (from, to_player, t0, dur)
        for s, e, st in segs:
            if st in (0, 1):
                team_off = st * 11
                if not (team_off <= carrier < team_off + 11):
                    # turnover: nearest player of the new team takes the ball
                    d = np.hypot(*(cur[team_off:team_off + 11] - b).T)
                    carrier = team_off + int(np.argmin(d))
                next_pass = s + int(rng.uniform(1.5, 3.0) * FPS)
            for t in range(s, e + 1):
                for k in range(2):
                    sl = slice(k * 11, (k + 1) * 11)
                    shift = 0.0
                    if st == k:
                        shift = 10.0
                    elif st in (0, 1):
                        shift = -5.0
                    tgt_att = form + [shift, 0.0]
                    tgt = to_raw(tgt_att, k, half)
                    if st in (0, 1):
                        tgt = tgt + 0.25 * (b - tgt)
                    vmax = 6.0 if st != 2 else 1.2
                    step = tgt - cur[sl]
                    n = np.hypot(*step.T)[:, None] + 1e-9
                    cur[sl] += step / n * np.minimum(n, vmax / FPS) + rng.normal(0, 0.03, (11, 2))
                if st in (0, 1):
                    if flight is None and t >= next_pass:
                        team_off = st * 11
                        mates = [p for p in range(team_off, team_off + 11) if p != carrier]
                        to = int(rng.choice(mates))
                        dist = np.hypot(*(cur[to] - b))
                        flight = (b.copy(), to, t, max(3, int(dist / 15.0 * FPS)))
                        next_pass = t + flight[3] + int(rng.uniform(1.5, 3.0) * FPS)
                    if flight is not None:
                        b0, to, t0, dur = flight
                        a = (t - t0) / dur
                        b = b0 + (cur[to] - b0) * min(a, 1.0)
                        span = np.hypot(*(cur[to] - b0))
                        if span > 25:                          # long balls go in the air
                            height[t] = 4 * min(span / 5, 8.0) * min(a, 1) * (1 - min(a, 1))
                        if a >= 1.0:
                            carrier, flight = to, None
                    else:
                        cur[carrier] += 0.15 * (b - cur[carrier])
                        b = cur[carrier] + [0.4, 0.0]
                else:
                    flight = None
                    if t == s:
                        out_dir = 1.0 if b[1] > W / 2 else -1.0
                    if -2.0 < b[1] < W + 2.0 and t - s < 120:  # roll out over the touchline
                        b = b + [0.0, out_dir * 0.6]
                pos[t] = cur
                ball[t] = b
    return rng, state, pos, ball, height


def tracker(rng, state, pos, ball, height):
    """Turn true positions into what an automatic tracker reports: slots, dropouts, NaN ball."""
    players = np.full((N, N_HOME_SLOTS + N_AWAY_SLOTS, 3), np.nan, dtype=np.float32)
    home_slots, away_slots = team_slots()
    gk_home, gk_away = home_slots[-1], away_slots[-1]
    slot_of = np.concatenate([[gk_home], home_slots[:10], [gk_away], away_slots[:10]])
    missing = np.zeros(22, dtype=bool)
    dup = None                                             # (player, slot, until)
    in_match = state != 3
    prev = pos[0].copy()
    for t in range(N):
        if not in_match[t]:
            continue
        if rng.random() < 0.003 and t > 0:                 # tracker stall: frame repeated
            players[t] = players[t - 1]
            continue
        sw = rng.random(22) < 1 / 750                      # identity switch to a free slot
        sw[[0, 11]] = False                                # goalkeepers keep their slots
        for p in np.flatnonzero(sw):
            pool = (home_slots[:-1] if p < 11 else away_slots[:-1])
            free = np.setdiff1d(pool, slot_of[(0 if p < 11 else 11):(11 if p < 11 else 22)])
            if len(free):
                slot_of[p] = rng.choice(free)
        flip = np.where(missing, rng.random(22) < 0.12, rng.random(22) < 0.009)
        missing ^= flip
        spd = np.hypot(*(pos[t] - prev).T) * FPS
        prev = pos[t]
        for p in range(22):
            if not missing[p]:
                sp = spd[p] if (p not in (0, 11) or rng.random() < 0.2) else np.nan
                players[t, slot_of[p]] = (pos[t, p, 0] / L, pos[t, p, 1] / W, sp)
        if (t // 4000) % 4 == 0 and np.isnan(players[t, home_slots[22], 0]):
            players[t, home_slots[22]] = (0.5, -0.065, 0.3)   # someone on the bench side: far off
        if state[t] == 2 and np.isfinite(ball[t, 1]) and not (0 <= ball[t, 1] <= W) \
                and np.isnan(players[t, away_slots[20], 0]):   # throw-in taker just off the pitch
            players[t, away_slots[20]] = (ball[t, 0] / L, np.clip(ball[t, 1], -1.2, W + 1.2) / W, 0.5)
        if dup is None and rng.random() < 1 / 2000:        # same person tracked twice
            p = int(rng.integers(1, 11))
            pool = home_slots[:-1]
            free = np.setdiff1d(pool, slot_of[:11])
            if len(free):
                dup = (p, int(free[0]), t + int(rng.uniform(50, 125)))
        if dup is not None:
            p, sl, until = dup
            players[t, sl] = (pos[t, p, 0] / L + 0.002, pos[t, p, 1] / W, spd[p])
            if t >= until:
                dup = None
    # ball visibility: two-state Markov chain, mostly hidden in dead time
    vis = np.zeros(N, dtype=bool)
    v = True
    for t in range(N):
        if not in_match[t]:
            v = False
            continue
        # real data: few short gaps, missing ball comes in long stretches (p50 gap ~ many s)
        p_lose, p_find = (0.0022, 0.0035) if state[t] != 2 else (0.004, 0.003)
        v = (rng.random() >= p_lose) if v else (rng.random() < p_find)
        vis[t] = v
    # the real ball track is smoothed by the tracker: a kick's velocity change is spread over
    # ~0.5 s. Reproduce that with a Gaussian smoother (sigma 5 frames) on the true trajectory.
    kern = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)
    kern /= kern.sum()
    ball = ball.copy()
    for c in range(2):
        col = ball[:, c].astype(np.float64)
        ok = np.isfinite(col)
        num = np.convolve(np.where(ok, col, 0.0), kern, mode="same")
        den = np.convolve(ok.astype(np.float64), kern, mode="same")
        ball[:, c] = np.where(ok, num / np.maximum(den, 1e-9), np.nan)
    balls = np.full((N, 3), np.nan, dtype=np.float32)
    balls[vis, 0] = ball[vis, 0] / L
    balls[vis, 1] = ball[vis, 1] / W
    hv = vis & np.isfinite(height) & (height > 0.3)
    balls[hv, 2] = height[hv]                              # third channel = height, mostly NaN
    # a few tracker glitches: single-frame teleports
    glitch = np.flatnonzero(vis)[rng.integers(0, vis.sum(), 200)]
    balls[glitch, 0] = rng.random(200)
    return players, balls


def write_meta(path):
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<main>", "  <Metadata>",
             "    <GlobalConfig><FrameRate>25</FrameRate>",
             "      <TrackingType>Optical</TrackingType>",
             "      <FieldSize><Width>105</Width><Height>68</Height></FieldSize>",
             "    </GlobalConfig>",
             "    <Teams>",
             '      <Team id="FIFATMA"><Name>Barcelona</Name></Team>',
             '      <Team id="FIFATMB"><Name>Atletic Club</Name></Team>',
             "    </Teams>", "    <Players>"]
    n = N_HOME_SLOTS + N_AWAY_SLOTS
    for i in range(n):
        team = "FIFATMA" if i < N_HOME_SLOTS else "FIFATMB"
        pos_type = "Goalkeeper" if i + 1 in (N_HOME_SLOTS, N_HOME_SLOTS + N_AWAY_SLOTS) else "Unknown"
        lines.append('      <Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
                     "<ShirtNumber>{0}</ShirtNumber><ProviderPlayerParameters><ProviderParameter>"
                     "<Name>position_type</Name><Value>{2}</Value></ProviderParameter>"
                     "</ProviderPlayerParameters></Player>".format(i + 1, team, pos_type))
    lines += ["    </Players>", "    <Devices><Device><Sensors><Sensor><Channels>",
              '      <Channel id="x"/><Channel id="y"/><Channel id="speed"/>',
              "    </Channels></Sensor></Sensors></Device></Devices>", "  </Metadata>",
              "  <DataFormatSpecifications>",
              '    <DataFormatSpecification startFrame="1" endFrame="{}">'.format(N),
              '      <StringRegister name="frameCount"/>',
              '      <SplitRegister separator=":">',
              '        <SplitRegister separator=";">']
    for i in range(n):
        lines.append('          <SplitRegister separator=",">'
                     '<PlayerChannelRef playerChannelId="player{0}_x"/>'
                     '<PlayerChannelRef playerChannelId="player{0}_y"/>'
                     '<PlayerChannelRef playerChannelId="player{0}_speed"/>'
                     "</SplitRegister>".format(i + 1))
    lines += ["        </SplitRegister>",
              '        <SplitRegister separator=",">'
              '<BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
              '<BallChannelRef channelId="speed"/></SplitRegister>',
              "      </SplitRegister>", "    </DataFormatSpecification>",
              "  </DataFormatSpecifications>", "  <PlayerChannels>"]
    for i in range(n):
        for c in ("x", "y", "speed"):
            lines.append('    <PlayerChannel id="player{0}_{1}" playerId="P{0}" channelId="{1}"/>'
                         .format(i + 1, c))
    lines += ["  </PlayerChannels>", "</main>"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_homography(path, rng, state, ball):
    """Camera that follows the (true) ball with lag; pitch (0-1) -> pixels, 40 m wide view."""
    cx = np.empty(N)
    cy = np.empty(N)
    c = np.array([0.5, 0.5])
    for t in range(N):
        if state[t] in (0, 1, 2) and np.isfinite(ball[t, 0]):
            target = np.array([ball[t, 0] / L, ball[t, 1] / W])
            c = c + (0.04 if state[t] != 2 else 0.01) * (target - c)
        cx[t], cy[t] = c
    sx = 1920 * L / 40.0
    sy = 1080 * W / 22.5
    with open(path, "w") as fh:
        fh.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        err = rng.gamma(2.0, 0.12, N)
        for t in range(N):
            fh.write("{},{:.4f},0,{:.4f},0,{:.4f},{:.4f},0,0,1,{:.4f},1\n".format(
                t, sx, 960 - sx * cx[t], sy, 540 - sy * cy[t], err[t]))


def fmt(v):
    return "NaN" if v != v else "{:.5f}".format(v)


def write_raw(path, players, balls):
    with open(path, "w") as fh:
        for t in range(N):
            groups = ";".join(",".join(fmt(v) for v in players[t, s]) for s in range(players.shape[1]))
            fh.write("{}:{}:{}\n".format(t + 1, groups, ",".join(fmt(v) for v in balls[t])))


def write_tagging(path, rng, state):
    names = {0: "Barcelona", 1: "Atletic Club"}
    inst = []

    def add(s, e, code, labels):
        inst.append((s / FPS, e / FPS, code, labels))

    blocks = half_video_blocks(GAME)
    add(blocks[0][1] - 60, blocks[0][1] + 65, "SET PIECES",
        [("Team", "Barcelona"), ("Half", "1st Half"), ("Type", "Kick Off (Start)")])
    add(KICKOFF_2ND - 60, KICKOFF_2ND + 65, "SET PIECES",
        [("Team", "Atletic Club"), ("Half", "2nd Half"), ("Type", "Kick Off (Start)")])
    # restarts and phases from the simulator's segments
    change = np.flatnonzero(np.diff(state.astype(np.int16)) != 0) + 1
    bounds = np.r_[0, change, len(state)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        st = int(state[s])
        half = "1st Half" if s < 100000 else "2nd Half"
        if st in (0, 1) and s > 0 and state[s - 1] == 2 and rng.random() < 0.85:
            kind = rng.choice(["Throw-in", "Corner Kick", "Goal Kick", "Free Kick"],
                              p=[0.5, 0.2, 0.2, 0.1])
            team = "N/A" if kind == "Throw-in" else names[st]
            add(s - 40, s + 60, "SET PIECES", [("Team", team), ("Half", half), ("Type", str(kind))])
        if st in (0, 1) and e - s > 5 * FPS and rng.random() < 0.7:
            add(s + 10, e - 10, rng.choice(["BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD"]),
                [("Team", names[st]), ("Half", half)])
            add(s + 20, e, "DEFENDING IN MIDDLE THIRD", [("Team", names[1 - st]), ("Half", half)])
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<file>", "  <ALL_INSTANCES>"]
    for i, (s, e, code, labels) in enumerate(sorted(inst, key=lambda r: r[0])):
        lab = "".join("<label><group>{}</group><text>{}</text></label>".format(g, t)
                      for g, t in labels)
        lines.append("    <instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end>"
                     "<code>{}</code>{}</instance>".format(i + 1, s, e, code, lab))
    lines += ["  </ALL_INSTANCES>", "</file>"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="bundle root; tactical/ is created inside")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    folder = os.path.join(args.out, "tactical")
    os.makedirs(folder, exist_ok=True)
    rng, state, pos, ball, height = simulate(args.seed)
    players, balls = tracker(rng, state, pos, ball, height)
    write_homography(os.path.join(folder, "BAR-ATH_tactical_homography.csv"), rng, state, ball)
    write_meta(os.path.join(folder, "BAR-ATH_tactical_atd.xml"))
    write_raw(os.path.join(folder, "BAR-ATH_tactical_atd.txt"), players, balls)
    write_tagging(os.path.join(folder, "BAR-ATH_tactical_smart_tagging.xml"), rng, state)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "out", "synthetic_truth.npz"), state=state)
    with open(os.path.join(args.out, "synthetic_gt.csv"), "w") as fh:   # same format as the public GT
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(t, "HADX"[state[t]]) for t in range(N))
    graded = state != 3
    print("synthetic bundle written to", folder)
    print("ball visible: {:.1f}% of all, {:.1f}% of in-match frames".format(
        100 * np.isfinite(balls[:, 0]).mean(), 100 * np.isfinite(balls[graded, 0]).mean()))
    print("tracked objects per in-match frame: {:.1f}".format(
        np.isfinite(players[graded, :, 0]).sum(1).mean()))
