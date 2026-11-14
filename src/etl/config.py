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
# The ball's third channel is labelled a speed in the XML but behaves like height in metres (no
# correlation with speed from positions, 0-7 m, present ~10% of frames). Treated as height; the
# report prints the evidence (its correlation with ball speed) on every build.
BALL_AERIAL_Z = 2.0            # m; above head height the ball cannot be under foot control

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

# ---- tracking hygiene -------------------------------------------------------------------------
DUP_RADIUS = 1.0               # m; two same-team tracks closer than this are one person tracked twice
OFF_PITCH_M = 0.5              # m outside a line: "off the pitch" (a throw-in or corner taker);
                               # kept for distance to the ball, excluded from team shape
FAR_OFF_M = 3.0                # m outside: not a participant (bench, warm-up, staff); dropped
SLOT_VEL_K = 5                 # frames each side for per-track velocity
SLOT_VEL_MAX = 12.0            # m/s; faster implied motion inside one slot = identity switch

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
