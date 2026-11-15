# The ETL

One deterministic build turns the raw tactical bundle into tables that every later step (dead-ball
detection, control model, segmentation) reads in seconds instead of re-parsing 187 MB of text.

```bash
pip install -r requirements-etl.txt
python tests/selftest.py          # ~1 min, synthetic data only: proves the build runs and is deterministic
python -m src.etl.build           # ~1 min on the real data; writes cache/ and prints cache/report.txt
python -m src.etl.verify_public   # what the signals mean against the public 5 minutes (aggregates only)
```

`verify_public` is a semantics check, in the sense CHALLENGE.md intends ("the public ground truth
is where you can see it"): are frames aligned, what do "dead" and "owning" look like in the data.
It is not a tuning tool, it writes nothing, and no label ever enters the ETL tables.

**Nothing in `cache/` may be committed or shared.** Frame-level exports reconstruct the licensed
data. `cache/report.txt` is the exception: aggregate statistics and checks only, written to be safe
to share.

**No ground truth is read anywhere in `src/etl/` except `verify_public.py`**, which joins labels
read-only for aggregate statistics and never writes them into ETL tables.

## Pipeline

| stage | module | what it does |
|---|---|---|
| 1 | `io_atd.py` | XML → team of each slot, channel layout (from the DataFormatSpecification, else `<Player>` order) and any ball-state words; `.txt` → arrays in one vectorised pass (~3 s) |
| 2–3 | `orient.py` | timeline (half, phase, graded, seconds into half); metres; **home attacks +x in both halves**, decided from the data per half (where each side's extreme players sit), with a margin in the report |
| 2b | `io_homography.py` | camera: where the image centre lands on the pitch, zoom, pan speed, which players and whether the ball are on screen |
| 4 | `ball.py` | removes single-frame teleports (> 42 m/s in and out), bridges gaps ≤ 10 frames inside a half unless the implied speed is impossible, centred velocity; every change is flagged |
| 5a | `players.py` | per position: **measured** (the tracker's speed present) or guessed; per-frame team with the keeper-slot correction; per-frame keeper role; same-team duplicates (measured kept over guesses); opposite-team one-object pairs; far-off-pitch people dropped |
| 5–7 | `features.py` | team shape/tempo (all kept positions), ball proximity / contest / contacts / control (**measured players only**; a contact needs the ball to react), dead-ball geometry |
| 7b | `situations.py` | dead-ball cues on graded frames: 2nd half before kickoff, sustained > 11 of one team, tracking blackouts, dense mixed clusters (set-piece crowds), crowds (confrontations, as a share of the tracked players), cooling breaks, ball stationary for seconds, ball isolated |
| 8 | `ball.py` | event candidates: multi-scale impulses, contact episodes, exits/entries, lost/found |
| 9 | `io_tagging.py` | Smart Tagging → frame intervals (team → H / A / N/A) and per-frame columns |
| 10 | `features.py` | 1 s and 3 s rolling means, centred (`__c`) and trailing (`__b`) |
| 11–12 | `checks.py`, `report.py`, `build.py` | checks against DATA.md numbers; aggregate report with corrections, situations and an invariant sweep (each with video timestamps to check), and how much the nearest-team answer changed against the previous build; `build.py` only orchestrates load → transform → write → check → report |

Every correction and cue follows a decision recorded in [PLAN.md](PLAN.md): data is corrected only
when the situation is physically impossible or a tracker behaviour confirmed on video; anything
merely unusual in football is reported or used as evidence, never corrected.

Shared helpers (run-length encoding, pitch distances) live in `util.py`. Every constant is in
`config.py`. None was fitted to any labels: they are published facts or physical priors, and the
manifest records them for every build.

## Team encodings

Two conventions are stored as-is (changing them would break content hashes):

| convention | values | where |
|---|---|---|
| **signed** (`TEAM_SIGN_*`) | +1 home · −1 away · 0 none / N/A | `frames.parquet` contest/contact/tag team columns |
| **index** (`TEAM_IDX_*`) | 0 home · 1 away · −1 nobody | `events.parquet` `team` / `nearest_team`, `slot_team.npy` |

## Outputs

| file | content |
|---|---|
| `frames.parquet` | 199,319 rows, `frame` = submission row, ~228 columns (below) |
| `events.parquet` | one row per event: `kind` ∈ impulse · contact · exit · entry · ball_lost · ball_found; team columns use the **index** convention |
| `ball_runs.parquet` | one row per continuous stretch of observed ball in a graded half |
| `tagging.parquet` | Smart Tagging intervals: `code, start_frame, end_frame` (half-open), `side` (H / A / N/A), `type`, all tags as JSON |
| `players_m.npy` | `[199319, 46, 3]` oriented metres + speed, as delivered |
| `players_meas.npy` | `[199319, 46]` bool: position measured (speed present), not guessed |
| `players_team.npy` | `[199319, 46]` int8 per-frame team after corrections (0 home · 1 away · −1 dropped) |
| `players_keeper.npy` | `[199319, 46]` bool: per-frame goalkeeper role |
| `ball_m.npy`, `slot_team.npy` | oriented raw ball; slot teams (**index** convention) |
| `manifest.json` | input sha256, parameters, output content hashes, column lists, checks |
| `report.txt` | aggregate statistics and checks; safe to share |

## `frames.parquet` columns

Prefixes: `h_` home, `a_` away. Team positions are in that team's own attacking frame
(`x_att`: 0 = own goal line, 105.3 = opponent's). Signed contest columns use the **signed**
convention and are **positive when they favour home**.

| group | columns |
|---|---|
| timeline | `frame, half (0/1/2), phase (0 pre · 1 H1 · 2 HT · 3 H2 · 4 post), graded, t_half_s, rotated` |
| ball | `ball_x, ball_y` (cleaned and bridged), `ball_observed, ball_interpolated, ball_jump_flag, ball_frames_since_seen, ball_frames_until_seen, ball_vx, ball_vy, ball_speed`, `ball_ch3` (raw third channel, meaning unknown, not a feature) |
| ball, around gaps | `ball_last_x, ball_last_y, ball_next_x, ball_next_y, ball_unreliable, ball_run_id` |
| collective movement | `flow_x, h_vx_att, a_vx_att` |
| off the pitch | `n_off_pitch, h_n_off_pitch, a_n_off_pitch, off_d1_ball, off_near_team` |
| camera | `cam_cx, cam_cy, cam_d_ball, cam_view_width_m, cam_pan_speed, cam_reproj_error, n_on_screen, frac_tracked_off_screen, ball_on_screen` |
| hygiene | `frozen_frame, h_n_dupes, a_n_dupes` |
| goalkeeper (×2) | `gk_x_att, gk_d_ball` (per-frame role, not slot) |
| team shape (×2, outfield) | `n_tracked, n_measured, n_outfield, cx_att, cy, spread, deepest_att, highest_att, length, width, own_half_frac, cx_att_vel` |
| team tempo (×2) | `mean_speed, max_speed, nearest_speed` |
| team vs ball (×2, measured only) | `d1_ball, d2_ball, n_within_3m, n_within_5m, n_within_crowd` |
| contact / control | `contact_team, control_team, last_contact_team, last_control_team, frames_since_contact, frames_since_control` |
| contest | `d1_diff, closer_team, contested_pair, closer_run, d1_min` |
| situations | `known_dead, h_surplus, a_surplus, track_blackout, dense_cluster, dense_cluster_n, crowd, crowd_n, crowd_share, cooling_break, ball_still_long, ball_isolated` |
| dead-ball cues | `n_tracked, all_mean_speed, speed_ratio_1s_10s, kickoff_shape, ball_oob, ball_out_dist, ball_d_touchline, ball_d_goalline, ball_d_centre, ball_stationary` |
| tagging | `tag_<code>`, `tag_<code>_team` for all 14 codes; `tag_vote_home`; set-piece / goal timing columns |
| context | `<col>__c1s, __b1s, __c3s, __b3s` for the columns listed in `config.CONTEXT_COLUMNS` |

Set-piece type codes: 0 throw-in · 1 corner kick · 2 goal kick · 3 free kick · 4 kick off (after
goal) · 5 kick off (start) · 6 other · −1 none.

## Known limitations

- More than 11 tracks for one team can remain; this is never capped (goals, confrontations,
  substitutions put more people on the pitch). Sustained surplus of measured players is a cue.
- Guessed positions (≈10% of all) are kept for team shape; they never touch the ball features.
- The keeper-slot correction relies on the tracker's measured flag; a keeper the tracker never
  measures keeps its slot's team (and is dropped only if it sits in the opponent's half).
- `nearest_team` and `closer_team` are only as good as the ball position (~41% of graded frames
  missing).
- Impulse detection only runs where the ball was observed on both sides of the frame.
- `last_control_team` cannot see a change of possession inside a long missing-ball gap.
- The camera centre is a proxy for where play is; its value is in stretches without a ball.
- Orientation assumes each team's extreme players sit nearest the goal they defend over a half.
