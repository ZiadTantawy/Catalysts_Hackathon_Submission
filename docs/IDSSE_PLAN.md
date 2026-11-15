# IDSSE training track: strategy and plan

Team Catalysts · More than a Hack 2027, Phase 1 · written 2026-11-15 · deadline 2026-11-17

Half the team trains on the IDSSE dataset (DFL Bundesliga, public) and delivers per-frame evidence
that the other half's labeller (etl-v1.3 → dead-ball detector → semi-Markov decoder) consumes on
BAR-ATH. This document is the working agreement for that track.

## Why IDSSE, and what it is for

1. **Its labels mean the same thing as ours.** Every ball frame in the IDSSE positions XML carries
   `BallStatus` (alive / dead) and `BallPossession` (team). CHALLENGE.md says the BAR-ATH ground truth
   is derived the same way: `D` where corrected tracking marks the ball dead, otherwise the team the
   tracking names. IDSSE is the closest labelled proxy we can get.
2. **It attacks the bottleneck.** The fork scan's oracle analyses: a perfect team signal gives
   spellF1 .88–.95, a perfect dead mask only .52–.63. Our dead-ball detector is already rule-based
   and checked on video. **The IDSSE track's main deliverable is the team signal during live play,
   especially inside missing-ball gaps.**
3. **It gives us a real validation set.** Right now the only labels we have are the 5 public minutes,
   which we may not tune on. A labelled, degraded IDSSE bench lets us choose decoder costs, minimum
   durations and model versions honestly. **That alone is worth the effort, even if the model
   transfers badly** (see the fallback at gate G2).

Allowed by RULES.md: external data about other games, human-produced or not, provided the organisers
can access it (public figshare DOI) and the technical document explains it.

## What is on disk (checked 2026-11-15)

`external-data/idsse/` (also `idsse.zip`, same content):

| match | matchinformation | events | positions | usable |
|---|---|---|---|---|
| J03WMX | yes | yes | 419 MB | yes |
| J03WN1 | yes | yes | 392 MB | yes |
| J03WOH | yes | yes | 349 MB | yes |
| J03WOY | yes | yes | 363 MB | yes |
| J03WPY | yes | **S3 "Request has expired" error page** | **35 MB, truncated** | no |
| J03WQQ | yes | yes | **missing** | no |
| J03WR9 | yes | yes | **missing** | no |

**T0 done (2026-11-15):** `src/idsse/download.py` (section 0 of `eda/idsse_01_parse_labels.ipynb`)
repaired WPY, WQQ and WR9 from figshare (article 28196177), MD5-checked. All 7 matches usable.

Format: 25 Hz (same as BAR-ATH), metres, origin at the centre spot, 105 × 68 m, one `<FrameSet>` per
person per half; the ball FrameSet adds `Z`, `BallPossession` and `BallStatus`. **To verify in T1:**
which `BallPossession` value is home (expected 1 = home, 2 = away; check against matchinformation and
kickoff), and what `M` means.

## The core problem: domain gap

IDSSE is clean corrected tracking: 22 identified players and the ball on every frame. BAR-ATH ATD is
automatic: ball missing on 41% of graded frames (in long gaps, mostly during dead ball and slow play),
~10% of player positions guessed, 46 anonymous slots, keeper slots carrying the wrong team. A model
trained on clean IDSSE learns "look at the ball" and fails exactly where we need it.

So the plan rests on three things:

1. **One feature code path.** IDSSE goes through the same etl-v1.3 code as BAR-ATH. Never write a
   second feature implementation.
2. **A degrader** that makes IDSSE look like ATD, calibrated on label-free BAR-ATH statistics.
3. **A bench** that checks the degrader against published numbers before any model is trusted.

## Workstreams

### W1 · IDSSE → ATD-shaped arrays (adapter)

`build.py` is array-based from stage 3: it needs `raw_players [N, 46, 3]` (normalised 0–1 x, y, speed;
speed NaN = guessed), `ball_raw [N, 3]`, `slot_team`, `slot_is_gk`, a timeline and camera features.

- Stream-parse the positions XML (line regex or `lxml.iterparse`; 400 MB each, do not DOM-load), cache
  to `.npz` per match. Players → slots, ball, `Z`, labels per frame.
- **Refactor needed in etl-v1.3:** `C.N_FRAMES` and the graded half blocks are constants for BAR-ATH.
  Parameterise `orient.timeline()` and the stages that use `N_FRAMES` with `(n_frames, half_blocks)`;
  BAR-ATH output must stay byte-identical (the manifest content hashes prove it).
- No Smart Tagging and no homography in IDSSE → **tagging features are excluded from the model**; camera
  features come from the virtual camera in W2.

### W2 · Degrader (the part that decides whether transfer works)

Every rule is physical and calibrated to BAR-ATH aggregates in `cache/report.txt` (no labels):

| ATD behaviour | Simulation on IDSSE | Calibration target (BAR-ATH, label-free) |
|---|---|---|
| Camera follows play, ~62 m wide | Virtual camera: centre = smoothed mix of ball and team blocks, width ~62 m | view width p50 62.4 m; camera-ball distance p50 23.6 m |
| Off-screen players are guessed | Outside the view: freeze or drift the position, speed = NaN | measured 90.2% of positions; 5.0% tracked off screen |
| Ball lost in the air, in crowds, when out | Drop the ball when `Z` high, many players within 3 m, off-screen, out of pitch, or slow dead phases; state-dependent rates | visible 59.3% of graded frames; 150 gaps, p50 / p90 10.7 / 33.8 s; 99.7% of missing frames in gaps > 5 s |
| Track fragmentation and duplicates | Re-assign identities to anonymous slots on occlusion; occasional duplicate | ~20.2 objects tracked; > 11 on the pitch on 8.8% (home) / 3.5% (away) of graded frames |
| Keeper-slot errors | Some keeper-slot tracklets carry an outfield player of the other team | P6 correction counts |
| Provider-smoothed ball, drift | Light smoothing + position noise ~0.5 m | 2nd difference p50 1.1 cm |
| **Ball keeps moving when dead** | Do not freeze the ball during `BallStatus = 0` | fork finding 4 |

**Calibration checks, in this order** (all on published numbers, not on the public window):

1. Marginals: the table above, IDSSE-degraded vs BAR-ATH.
2. **Baseline transfer:** run `starter/baseline.py` (smooth 25/50/125, raw) on degraded IDSSE. The
   published full-match numbers are smooth50 spellF1 **.243**, macroF1 ~.65–.68, with ranking
   smooth50 > smooth125 > smooth25 > raw on spellF1. If degraded IDSSE gives ~.24 and the same ranking,
   the degrader is hard enough. Clean IDSSE will score far higher; that gap is what we are closing.
3. **Adversarial validation:** a classifier separating BAR-ATH frames from degraded-IDSSE frames on
   the model features. Features it uses most are where the degrader is wrong; fix them or drop them.

### W3 · Labels

- `D` = `BallStatus` dead; `H` / `A` = `BallPossession` while alive. Orient home attacking +x, as the ETL.
- **Relabel to controlled possession.** DFL possession switches about twice as often as the official
  GT (fork finding 3; several teams' biggest single gain). Give back spells shorter than τ when
  possession returns to the previous team. Choose τ (1.5–3 s range) from football priors and the
  published GT statistics (mean possession 14.2 s, shortest spell 0.76 s, DATA.md), never by scoring
  the public window.
- Variant B if time: change possession only at a DFL event showing control (successful pass /
  reception / ball claim by the new team). Compare A vs B on the IDSSE bench.

### W4 · Models

Two heads, matched to the existing labeller:

| head | target | input | role in the decoder |
|---|---|---|---|
| **team** (priority) | P(home in possession \| live) | ETL features, weighted towards signals that survive a missing ball: team shape, block movement, `flow_x`, last/next contact team, time since contact, context windows past/next 1 / 5 / 15 s | team emission, decisive inside ball gaps |
| **turnover** | P(possession changes within ±0.5 s) | same + contact episodes | transition placement (spellF1 is unforgiving: 1 s late = .65) |
| dead (optional) | P(dead) | dead-ball cues | an extra evidence term in `deadball.py`, weighted against the rules, never replacing them |

- Model: **LightGBM / HistGBM first** (Rushilkr77: GBM + Viterbi, .594 public / .664 own bench, no
  tuning on public). A BiGRU/TCN only if the GBM is done and the bench shows headroom.
- **Augmentation:** swap home/away (mirror x, swap team features, invert label) and mirror y. Doubles
  the data and removes home bias; the team head should be antisymmetric.
- Training data: degraded IDSSE at **several severities** (light / ATD-like / heavy), so the model is
  not fit to one guess of the noise.
- Output to the other half: `cache/idsse_evidence.parquet` with `frame, logp_home, logp_away,
  p_turnover, p_dead` for all 199,319 BAR-ATH frames. **Agree this contract on day 1.**

### W5 · Bench and decoder tuning

- Leave-one-match-out on degraded IDSSE, scored with the official `evaluation/possession_metrics.py`
  (spellF1 primary, macroF1 tie-break), bootstrap intervals.
- Use it to choose: relabel τ, model version, decoder switching cost / minimum D duration / duration
  prior, X-abstention policy.
- **Robustness rule:** a choice is accepted only if it wins at ATD-like severity **and** does not lose
  at light and heavy severity. The degrader is our biggest unknown; this protects against it.
- Label-free sanity on BAR-ATH output: possessions per minute, number of stoppages (typical 100–130),
  ball in play (typical 52–58 min), home share.
- The public 5 minutes: run **once** on the final pipeline as a smoke test, as PLAN.md already agrees.

## Gates

| gate | when | pass | if it fails |
|---|---|---|---|
| G0 | 15 Nov evening | 4+ matches parsed, labels verified, ETL runs on IDSSE, BAR-ATH hashes unchanged | Drop the ETL refactor; build a thin feature set on IDSSE + BAR-ATH caches directly |
| G1 | 16 Nov 10:00 | Degraded smooth50 within ~.20–.30 spellF1 and baseline ranking matches the published one | Increase ball loss / camera window; do not train on an unchecked degrader |
| G2 | 16 Nov 14:00 | Team head + decoder beats the rule-based team evidence on the IDSSE bench (outside bootstrap noise) | **Fallback:** use IDSSE only to set decoder hyperparameters and to validate the rule-based labeller. Still a win |
| G3 | 16 Nov 20:00 | Evidence file delivered, decoder integrated, label-free priors plausible | Ship without the IDSSE head |
| Freeze | 17 Nov 12:00 | Clean clone regenerates the identical `Catalysts.csv` | |

## Timeline and split (two people on this track)

| when | Person A: data | Person B: model and bench |
|---|---|---|
| 15 Nov afternoon | T0 re-download; T1 parser + label checks; W1 ETL parameterisation | Evidence contract with the decoder half; bench harness on clean IDSSE (scorer, LOMO splits, baselines) |
| 15 Nov evening | W2 degrader v0, marginals report | W3 relabel (variant A); clean-IDSSE GBM to debug the pipeline → **G0** |
| 16 Nov morning | Degrader calibration, adversarial validation → **G1** | Team + turnover heads on degraded data, augmentation, severity sweep |
| 16 Nov afternoon | Help on features flagged by adversarial validation; variant B labels if time | **G2**; decoder hyperparameters on the bench; export evidence on BAR-ATH → **G3** |
| 17 Nov morning | Reproducibility: download script, determinism, fixed seeds | Technical-document section: data, degrader, labels, bench, results with intervals |

## Reproducibility and data rules

- Deterministic training: fixed seeds, LightGBM `deterministic=True`, fixed thread count. Also
  **commit the trained model file** (trained only on IDSSE) with its training script, so the submission
  regenerates exactly even if retraining drifts by a float.
- `scripts/get_idsse.py`: downloads IDSSE from figshare with checksums. IDSSE raw files and caches are
  not committed (size), BAR-ATH caches never (licence).
- Cite IDSSE (Bassek et al., Scientific Data 2025) and its DOI in the technical document.
- No BAR-ATH label touches training or selection. The public window is a smoke test only.

## Risks

| risk | likelihood | mitigation |
|---|---|---|
| Degrader does not match ATD, model does not transfer | high | G1 calibration, adversarial validation, severity sweep, G2 fallback |
| DFL possession semantics differ from Metrica's GT | medium | Controlled-possession relabel; variant B from events |
| Missing / broken IDSSE files | known | T0 re-download; 4 matches is enough to start |
| ETL refactor breaks BAR-ATH output | medium | Manifest content hashes must be identical before merging |
| Bundesliga vs LaLiga style | low | Features are relative (shape, movement), not absolute |
| Time | high | GBM only unless G2 passes early; the fallback still pays |
