"""Every constant the ETL uses, in one place.

None of these values were fitted to any ground truth. They are either published facts about the
data (frame counts, pitch size, half boundaries -- all from games.py / DATA.md) or physical and
footballing priors (how fast a ball can travel, how long a tracker gap is worth bridging). Changing
one changes the output, so they live here and are written into the manifest of every build.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

# ---- published facts --------------------------------------------------------------------------
FPS = GAMES[GAME]["fps"]                       # 25
N_FRAMES = GAMES[GAME]["n_video_frames"]       # 199319, the submission length
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]        # 105.3 x 68.0 m
HALF_BLOCKS = half_video_blocks(GAME)          # [(1, 11185, 90651), (2, 110808, 190197)]
# Published fact (FAQ.md): the second half is graded from 110,808 but kicks off at 113,742; the
# 2,934 frames between are all dead ball. The first half is graded from its kickoff.
KICKOFF_2ND = 113742

# Reference figures quoted in DATA.md, used only as build checks (never as features)
EXPECTED = {
    "ball_visible_all": 0.569,
    "ball_visible_graded": 0.593,
    "tracked_mean_graded": 20.5,
    "n_slots": 46,
    "kickoffs": {"1st Half": 11185, "2nd Half": 113742},
}

# ---- output -----------------------------------------------------------------------------------
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")

# ---- parsing ----------------------------------------------------------------------------------
PARSE_CHUNK = 4000                             # lines per vectorised parse

# ---- camera -----------------------------------------------------------------------------------
IMAGE_W, IMAGE_H = 1920, 1080

# ---- ball cleaning (physical priors) ----------------------------------------------------------
BALL_MAX_SPEED = 42.0          # m/s; hardest shots ~35-40 m/s. Faster implied motion = glitch
INTERP_MAX_GAP = 10            # frames (0.4 s); longer gaps are left NaN, never invented
VEL_HALF_WINDOW = 2            # frames each side for the centred ball-velocity estimate
# The ball's third channel is labelled a speed in the XML, does not behave like one, and is present
# even on frames with no ball position. Its meaning is unknown, so it is kept raw (ball_ch3) and
# not turned into any feature (docs/PLAN.md D6).

# ---- ball states (docs/PLAN.md D7): flagged, never deleted ------------------------------------
BALL_STILL_LONG = 75           # frames (3 s) of a stationary ball: a restart being set up
BALL_ISOLATED_R = 15.0         # m from every measured player ...
BALL_ISOLATED_SPEED = 2.0      # ... while slower than this: spare ball or false detection
BALL_ISOLATED_MIN = 13         # frames (0.5 s) sustained

# ---- events (kick / touch candidates) ---------------------------------------------------------
EVENT_SCALES = (3, 6, 12)      # frames before/after: 0.12 s, 0.24 s, 0.48 s. The ball track is
                               # smoothed, so a kick's velocity change is spread over many frames
EVENT_MIN_DV = 2.0             # m/s absolute floor for any impulse candidate
EVENT_Z_MIN = 3.0              # candidate: dv at least 3 robust noise scales above its median
EVENT_NOISE_K = 5.0            # "strong": 5 noise scales. Noise is estimated per scale from the
                               # data itself (median + MAD), so jitter does not become "touches"
EVENT_MIN_SEP = 8              # frames between two impulse candidates (non-maximum suppression)
EVENT_NEAR_R = 2.0             # m, "player at the ball" radius for event attribution
BALL_LOST_MIN_GAP = 25         # frames of invisibility before a lost/found event is emitted

# ---- contacts and control (the challenge's own proxy: "a player keeps the ball") --------------
CONTACT_R = 1.5                # m; ball this close to its nearest player = in contact
CONTACT_MIN = 3                # frames; shorter contact runs are noise
CONTACT_GAP_TOL = 2            # frames of interruption tolerated inside one contact run
CONTROL_MIN = 12               # frames (0.48 s) of continuous same-team contact = controlled
# A touch needs the ball to react (docs/PLAN.md D4): a ball flying past a player at speed is not a
# contact. A contact run is kept if the ball stays close for CONTACT_HOLD frames, or if the ball's
# velocity changes by at least CONTACT_REACT_DV across +-CONTACT_REACT_K frames inside the run.
CONTACT_HOLD = 6               # frames (0.24 s)
CONTACT_REACT_K = 3            # frames each side
CONTACT_REACT_DV = 2.0         # m/s

# ---- tracking hygiene -------------------------------------------------------------------------
DUP_RADIUS = 1.0               # m; two same-team tracks closer than this are one person tracked twice
OFF_PITCH_M = 0.5              # m outside a line: "off the pitch" (a throw-in or corner taker);
                               # kept for distance to the ball, excluded from team shape
FAR_OFF_M = 3.0                # m outside: not a participant (bench, warm-up, staff); dropped
SLOT_VEL_K = 5                 # frames each side for per-track velocity
SLOT_VEL_MAX = 12.0            # m/s; faster implied motion inside one slot = identity switch

# ---- players (docs/PLAN.md D1-D3, P6, P8, P9) -------------------------------------------------
# The ATD leaves a player's speed empty when the tracker did not see them and is holding or
# predicting the position: every off-screen position has no speed, every position with a speed is
# on screen. "Measured" = position and speed both present. Only measured players may be nearest to
# the ball, in contact or in control; guessed positions are used for coarse team shape only.
#
# Keeper slots (P6): the tracker gives a goalkeeper detection to either keeper slot, whatever the
# team (confirmed on video in both directions). A MEASURED keeper-slot track belongs to the team
# defending the goal it stands at, if it is within KEEPER_GOAL_ZONE of that goal line, is the
# deepest measured player towards it, and the penalty box holds fewer than KEEPER_BOX_CROWD other
# measured players (a keeper who has gone up for a late corner stands in a crowded box).
KEEPER_GOAL_ZONE = 25.0        # m from the goal line
KEEPER_BOX_CROWD = 4           # other measured players inside that penalty box
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16    # m, penalty area
# Per-frame keeper role (D3): the keeper-slot track if measured and on its own team's goal side,
# else the team's deepest measured player inside its own penalty area.
# Opposite-team duplicates (P9): the same object carrying both tags. Far tighter than marking or
# shirt-pulling: closer than XDUP_DIST, relative speed below XDUP_RELV, sustained XDUP_MIN frames.
# Never deleted (we cannot know which tag is wrong); ball attribution through such a pair is
# neutral (contested).
XDUP_DIST = 0.3                # m
XDUP_RELV = 0.3                # m/s
XDUP_MIN = 50                  # frames (2 s)
PLAYER_VEL_K = 2               # frames each side for per-track velocity in the duplicate test
PAIR_CHUNK = 5000              # frames per chunk for pairwise (46 x 46) computations

# ---- situations (docs/PLAN.md P7, P10, D8, D9): dead-ball cues, graded frames only ------------
SURPLUS_MIN = 75               # frames (3 s): more than 11 measured players of one team, sustained
BLACKOUT_RATIO = 0.5           # measured players below half of their 10 s rolling median ...
BLACKOUT_MIN = 5               # ... for at least this many frames
BLACKOUT_ABS = 10              # or fewer than this many measured players at all ...
BLACKOUT_ABS_MIN = 125         # ... for 5 s (a long loss the rolling median adapts to)
CLUSTER_R = 5.0                # m: dense mixed cluster = at least CLUSTER_N people within CLUSTER_R
CLUSTER_N = 8                  #    of one player,
CLUSTER_MINORITY = 0.25        #    the smaller team at least this share of them,
CLUSTER_SPEED = 1.5            #    their median speed below this (m/s),
CLUSTER_MIN = 125              #    sustained 5 s (set-piece crowds)
# Crowd (confrontations): looser than a set-piece crowd, and the tracker loses people inside it, so
# it is measured as a SHARE of the tracked players, which occlusion does not shrink. At least
# CROWD_SHARE of the tracked players within CROWD_R of one player, at least CROWD_MIN_TRACKED tracked,
# both teams (CLUSTER_MINORITY), slow (CLUSTER_SPEED), sustained CROWD_MIN. CROWD_SHARE sits above the
# 99th percentile reached in clearly live play over the whole match (0.63), with no labels involved.
CROWD_R = 10.0                 # m
CROWD_SHARE = 0.7
CROWD_MIN_TRACKED = 8
CROWD_MIN = 125                # frames (5 s)
COOL_SPEED = 1.0               # m/s, players' 3 s mean speed: a long stoppage ...
COOL_MAX_TRACKED = 12          # ... in which most players have disappeared from tracking ...
COOL_TOUCHLINE = 20.0          # ... and those still visible are within this of a touchline ...
COOL_MERGE = 375               # ... allowing 15 s interruptions ...
COOL_MIN = 1500                # ... lasting at least 60 s: a cooling / drinks break

# ---- dead-ball geometry -----------------------------------------------------------------------
OOB_MARGIN = 0.5               # m beyond a line before the ball counts as out
STATIONARY_SPEED = 0.6         # m/s
STATIONARY_WINDOW = 12         # frames of rolling mean before the stationary threshold
NEAR_RADII = (3.0, 5.0)        # m, players-near-ball counts
CROWD_RADIUS = 10.0            # m

# ---- context windows --------------------------------------------------------------------------
WINDOWS = {"1s": 25, "3s": 75}
SLOW_WINDOW = 250              # 10 s, the baseline for speed-drop ratios
CONTEXT_COLUMNS = [
    "d1_diff", "closer_team", "all_mean_speed", "ball_speed", "ball_observed",
    "h_cx_att_vel", "a_cx_att_vel", "h_cx_att", "a_cx_att", "kickoff_shape",
    "h_spread", "a_spread", "cam_pan_speed", "cam_cx", "n_tracked",
    "flow_x", "h_vx_att", "a_vx_att", "h_length", "a_length", "h_width", "a_width",
]

# ---- team encodings (two conventions; stored values must not change) --------------------------
# frames.parquet signed columns (+1 home / -1 away / 0 none or N/A)
TEAM_SIGN_HOME, TEAM_SIGN_AWAY, TEAM_SIGN_NONE = 1, -1, 0
# events.parquet team / nearest_team, and slot_team.npy (0 home / 1 away; -1 = nobody)
TEAM_IDX_HOME, TEAM_IDX_AWAY, TEAM_IDX_NONE = 0, 1, -1

# ---- Smart Tagging semantics ------------------------------------------------------------------
# Sign of each code's Team tag as evidence for "this team has the ball": +1 the tagged team is
# attacking, -1 the tagged team is defending (so the *other* team has it), 0 no possession meaning.
# This is a reading of OVERVIEW.md's code definitions, not something learned -- verify it on the
# public window before trusting it.
TAG_POSSESSION_SIGN = {
    "build up": +1, "progression": +1, "creating chances": +1, "attacking transition": +1,
    "ball in final third": +1, "ball in the box": +1, "players in the box": +1,
    "long ball": +1, "set pieces": +1, "goals": 0,
    "defending in attacking third": -1, "defending in middle third": -1,
    "defending in defensive third": -1, "defensive transition": -1,
}
SET_PIECE_CODE = "set pieces"
GOALS_CODE = "goals"
SET_PIECE_TYPES = ["throw-in", "corner kick", "goal kick", "free kick",
                   "kick off (after goal)", "kick off (start)"]
