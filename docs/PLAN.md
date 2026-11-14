# Plan and decisions

Team Catalysts · More than a Hack 2027, Phase 1 · last updated 2026-11-14 · deadline 2026-11-17

This is the working agreement for what we build next and why. Every rule below is generic: it is
evaluated on every frame of the match and fires wherever its condition holds. Nothing is tied to
specific frames of the game ([RULES.md](challenge/RULES.md)). Video was watched only to understand
and debug the data, never to produce labels or frame-specific corrections.

## Guiding principle

> The ETL may **correct** data only when the situation is **physically impossible** or is a
> **tracker behaviour confirmed on video**. Anything that is merely unusual in football (pressing,
> celebrations, duels, substitutions) is never corrected; at most it is reported, or used as a
> feature.

Corollaries:

- **Measured beats guessed.** The ATD leaves a player's speed empty when the tracker did not see
  them and is holding or predicting the position (100% of off-screen positions have no speed; every
  position with a speed is on screen). About 8% of player positions on graded frames are guesses.
- **No slot identities.** A slot is a tracklet, not a person; keeper slots carry other people.
- **Robust evidence over perfect cleaning.** Errors we have not found are absorbed by (a) features
  that do not depend on fragile signals and (b) a temporal model that needs consistent evidence
  over time to change possession.

## What the audit established

| Finding | Evidence (aggregates) |
|---|---|
| Ball track is already smoothed by the provider; no teleports, no jumps > 10 m | 2nd difference p50 1.1 cm; 0 single-frame spikes |
| Missing ball is mostly dead ball, not random | Mean player speed < 1.2 m/s on 66% of missing-ball frames vs 15% when visible |
| Speed channel = measured flag; the ETL currently overwrites it | 370,297 speeds filled from positions |
| Guessed positions cause most "> 11 players" | Home > 11: 13.6% of frames (all positions) vs 2.5% (measured only) |
| Keeper slots carry the wrong team at the wrong goal | Verified on video in both directions (see log below) |
| Nearest-player team changes on 2.9% of on-ball frames once measured-only + keeper fix are applied | 1,758 of 61,555 frames; 737 from keeper slots, 1,021 from guessed positions |
| Proximity alone counts a ball flying past as a touch | Seen at a back-pass where the ball passed 1.1 m from a player at 14 m/s |
| Public window is unrepresentative | Ball visible on 99.9% of its H frames vs 59% over the match |
| Ground truth flips possession near the new team's first contact; short spells exist | Most open-play changes within ~±0.6 s of a new-team contact; shortest spell 0.76 s |
| Most stoppages are not ball-out-of-bounds | 3 of 4 public-window stoppages had no ball out and no set-piece tag nearby |

## Decisions on data corrections (ETL v1.3)

| ID | Issue | Decision | Action | Status |
|---|---|---|---|---|
| D1 | Speed channel overwritten | Keep the tracker's own measured flag | Store `measured` per player position; stop filling speed into it (a separate derived speed column is fine) | Agreed |
| D2 | Guessed positions used near the ball | Only measured players can be nearest / in contact / in control | Guessed positions used only for coarse team shape, down-weighted | Agreed |
| P6 | Keeper slot at the wrong goal | **Correct** (tracker behaviour confirmed on video, both directions) | A measured keeper-slot track belongs to the team defending the goal it stands at, **only if**: within 25 m of that goal line, deepest measured player towards it, and fewer than 4 other measured players in that box (protects a keeper going up for a late corner) | Agreed: generalise with guards |
| D3 | Keeper role from slot numbers | Stop using slot-based keeper features | If needed, keeper = deepest measured player of a team near its own goal, per frame | Agreed |
| P5 | Player isolated among opponents | **Dropped** | High pressing does this by design; not an error | Agreed |
| P7 | More than 11 per team | **Never cap; evidence only when sustained** | Report only. As a dead-ball cue it counts only when it persists for several seconds (a 1 s spike is noise: a dead ball lasts until play restarts). It is one piece of evidence, never a decision on its own; the decoder's minimum dead-ball duration does the rest | Agreed |
| D9 | Player-cluster situations | **Dead-ball cues, graded frames only, sustained** | **Argument / confrontation**: one dense cluster of both teams (≥ 8 people within 5 m, both teams present, slow, ≥ 5 s) anywhere on the pitch. **Cooling break**: long stoppage (≥ 60 s) in which most players disappear from tracking and those still visible are near the bench-side touchline. Also: goal celebration (one-colour cluster, then kickoff shape), injury (few stationary players around a spot, long), substitution (one colour > 11 briefly near the halfway touchline). Note: the graded window is not exactly whistle to whistle. It starts 117 s before the 2nd-half kickoff and runs past the 1st-half whistle (players walk off inside graded time), so walk-offs and greetings can occur on graded frames; that is correct, because those frames are dead ball | Agreed. Cooling break: both candidates confirmed on video. Dense mixed cluster: all 4 candidates were set-piece crowds (2 free kicks, 2 corners), also dead ball, and 3 of the 4 had no Smart Tagging set-piece tag, so the rule recovers stoppages the tags miss. Kept as a general "dense mixed cluster" cue covering both set-piece crowds and confrontations |
| D10 | Confrontations (crowd) | **Separate flag, as a share of tracked players** | Confrontations are looser than set-piece crowds and the tracker loses people inside them, so a fixed count within 5 m misses them (end-of-match confrontation: median 6 within 5 m). Flag `crowd`: at least 70% of the tracked players (at least 8 tracked) within 10 m of one player, both teams, slow, for at least 5 s. 70% sits above the 99th percentile reached in clearly live play over the whole match (0.63), no labels involved. Widening the 5 m flag to 8 m was rejected: 40 stretches instead of 4, 8 of them with the ball moving. Continuous `crowd_n` and `crowd_share` are features. The 5 m `dense_cluster` flag is unchanged. Purely additive: all 297 existing columns and arrays verified identical | Agreed |
| P8 | Same-team duplicate tracks | Keep current de-duplication (< 1 m), on measured positions | Unchanged logic, measured-only | Agreed |
| P9 | Opposite-team tracks together | **Stricter rule, neutralise, never delete** | Duplicate only if < 0.3 m apart, velocity difference < 0.3 m/s, sustained ≥ 2 s. While such a pair is the nearest to the ball, ball attribution gives *no team evidence* (contested) | Agreed |
| P10 | Collapse in measured players | Keep, graded frames only | Flag as tracking blackout; also a dead-ball cue (close-ups after fouls). Extended in the build: also fewer than 10 measured players for ≥ 5 s, because a 10 s rolling median adapts to a long loss (a 65 s blackout at the end of the first half was otherwise 80% missed). Checked: never fires during clearly live play (35 of 35,232 live frames have < 10 measured, never for 5 s); blackouts are real absences, not a missing speed channel | Agreed |
| D4 | Contacts from proximity only | A touch needs the ball to react | Contact = measured player near the ball **and** the ball decelerates / changes direction or stays close for several frames | Agreed |
| D5 | Short gaps bridged at impossible speeds | Do not bridge | Bridge a gap ≤ 10 frames only if the implied speed across it is physically possible | Agreed |
| D6 | Ball third channel (meaning unknown) | Exclude from features | Keep the raw value in the cache, drop `ball_aerial` from modelling | Agreed |
| D7 | Ball states | Flag, never delete | Ball stationary ≥ 3 s, ball far from every measured player and slow, ball outside the pitch, ball faster than 42 m/s: each its own flag | Agreed |
| D8 | Second half graded before kickoff | Published fact | Frames from 110,808 to 113,741 are `D` (FAQ.md: 2,934 graded frames, all `D`); ungraded frames are `X` | Agreed |

## Diagnostics in every build report

The invariant sweep becomes part of `cache/report.txt`, so a new kind of error shows up as a count
rather than hiding. Each line: violation count, number of stretches, longest stretches with video
timestamps for checking.

Report-only (never correct): P4 measured player > 3 m off the pitch · P7 > 11 per team · P10
measured-player collapse · B1 ball > 42 m/s · B2 ball far from everyone and slow · B3 ball moving
fast outside the pitch · camera jumps and high reprojection error.

Correction counters: keeper-slot team changes (P6), opposite-team duplicate pairs neutralised (P9),
gaps not bridged (D5), and how often the nearest-team answer changed versus the previous build.

## The labeller (after ETL v1.3)

Approach: a three-state (`H`, `A`, `D`) semi-Markov model decoded with Viterbi over each half.

1. **Dead-ball evidence**: collective player speed collapse, ball missing / stationary / out,
   restart shapes (kick-off, set-piece positions), set-piece tags, measured-player collapse,
   unusual head counts, camera behaviour. Free kicks are tagged only 5 times in the match, so
   tracking cues must carry most of it.
2. **Team evidence**: measured contacts with ball reaction (D4), contested pairs neutral (P9);
   for frames without the ball, a classifier trained on algorithm-made pseudo-labels (confident,
   ball-visible frames) using ball-free signals: team shape, block movement, camera, Smart Tagging
   weighted by code reliability. Allowed by RULES.md (no person decides any pseudo-label).
3. **Decoding**: duration prior and switching costs so a few wrong frames cannot create a
   possession; transitions placed near the new team's first measured contact.
4. **Abstention**: `X` only inside long blind gaps where no change of possession is indicated
   (the scorer forward-fills `X` on the ranking metric). Never at a likely hand-over.

## Validation without tuning on the public window

- The public 5 minutes: one end-to-end smoke test only. Never used to choose between versions.
- Football priors: possession counts and durations, team shares, dead-ball share.
- Consistency: agreement with Smart Tagging codes that proved reliable; stability across halves.
- **Robustness stress test**: deliberately corrupt inputs (flip 5% of team tags, drop extra ball
  stretches, add duplicate tracks, jitter positions) and measure how much the output possessions
  move. Large movement points at the weakness to fix first.
- Optional if time allows: labelled tracking from other matches (allowed by RULES.md) as a real
  validation set.

## Schedule

| Day | Work |
|---|---|
| 14 Nov | Audit, video verification, this plan |
| 14 Nov | ETL v1.3 built on branch `etl-v1.3` (all checks pass, deterministic) |
| 15 Nov | Dead-ball detector |
| 16 Nov | Team evidence, decoder, first full submission; stress tests |
| 17 Nov | Technical document, README, clean-clone reproduction of the identical file, submit `Catalysts.csv` + `Catalysts.pdf` |

## Findings for the labeller

- **Flags are partial by design; long stoppages are covered by their combination.** Penalty argument to
  penalty (134 s): flags 25%, ball unseen 95%, players slow 96%, at least one sign 100%. Goal to
  cooling break (163 s): flags 50%, at least one sign 100%. But at least one sign is present on 52% of
  all graded frames, so the general signs must be weighed over time, never used as a rule.
- **Flags during clearly live play** (ball moving, players running): `dense_cluster`, `crowd`,
  `cooling_break`, `ball_still_long`, `ball_isolated` 0 frames; `track_blackout` 14; `h_/a_surplus`
  1,151 (~3.3%, leftover duplicates). P7 surplus is weak evidence and should carry little weight.
- **Goals and restarts**: Smart Tagging's `Kick Off (After Goal)` marks the restart after each goal (the
  ball is dead from the goal until then, ~3.5 minutes after the goal at 1:45:02). The `GOALS` interval
  starts before the goal (it includes the attack: 1:44:30 for the goal at 1:45:02), so its start is not
  the moment the ball went dead.
- Still to establish in the detector: exact dead-ball onset and restart frames, and short stoppages
  (throw-ins, quick free kicks); every stoppage checked so far lasted more than ~5 s.

## Open items

- README is still a skeleton; it must give the single command that regenerates the submission.
- Team must have 3–5 members for full eligibility.

## Video verification log

Watching to understand the data, not to label it. Times are on the tactical video.

| What was checked | Result |
|---|---|
| Nine stretches with slot 46 (Athletic keeper slot) near Barça's goal | Only Barça players there; slot 46 is a Barça player |
| Head count in one of them | 11 Barça, 10 Athletic (Athletic keeper off-screen) |
| Slot 46 holding the ball ~6.7 s near Barça's goal | Barça's keeper |
| Slot 46 receiving a back-pass ~11 m from Barça's goal | Consistent with Barça's keeper (deepest player, alone) |
| Slot 24 (Barça keeper slot) holding the ball ~10.6 s in front of Athletic's goal | Athletic's keeper |
| Second-half kickoff | Matches the published frame (113,742) |
| End of first half (P10 blackout 0:59:20–1:00:26) | Half over at 0:59:20, players leaving the pitch: dead ball, inside graded time |
| P9 near-duplicate at 0.5 m (slot 19 Barça + slot 34 Athletic): 1:30:54–1:30:57 | Corner set-up (players lining up from 1:30:43, corner taken at 1:31:03): two real players jostling, so the 0.3 m rule is right |
| End of match: when is the final whistle relative to the end of grading (2:06:47)? | Players gather from 2:06:07, confrontation clearly from 2:06:10 (about 12 players in it), final whistle exactly at 2:06:47, the end of grading. `crowd` covers 2:06:21–2:06:47 |
| `crowd` stretch 0:24:33–0:24:40 | Players lining up for a corner, taken at 0:24:40 (flag ends there) |
| `crowd` stretch 0:47:03–0:47:37 | Argument with the referee over a penalty, ball out of play; crowd separates around 0:47:37 (flag ends there) |
| `crowd` stretch 1:32:00–1:32:06 | Set piece being prepared, ball hidden behind a player, visible again 1:32:08–09 |
| `crowd` stretch 1:47:20–1:47:25 | Inside the cooling break after the goal at 1:45:02; players separate around 1:47:32 (`cooling_break` ends at 1:47:31) |
| Dense-cluster candidates (D9): 0:30:51, 0:38:01–0:38:18, 1:30:43, 1:37:33 | Free kick, corner, corner, free kick: all dead ball; only the first carried a set-piece tag |
| Cooling-break candidates (D9): 0:39:12–0:40:48 (clock ~31:44), 1:46:18–1:47:31 (clock ~75:28) | Both are cooling breaks |
