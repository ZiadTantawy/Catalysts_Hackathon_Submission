# IDSSE training results

Team Catalysts · run on Apple M5 (LightGBM/RF CPU, XGBoost CPU, PyTorch MPS) · finished 2026-11-16 00:32

Training data: 7 IDSSE matches, degraded at light / realistic / heavy, home↔away mirroring, leave-one-match-out on the 5 DEV matches. Decoder switching costs chosen on IDSSE DEV only (never on BAR-ATH or the public window).

## 1. IDSSE cross-validation (primary bench)

After hyperparameter search (LGBM 10 / XGB 12 / RF 6 trials) and final refits:

| Model | spellF1 | macroF1 | team-right (live) | logloss |
|---|---:|---:|---:|---:|
| LightGBM | **0.681** | 0.886 | 0.842 | 0.300 |
| XGBoost | 0.675 | 0.880 | 0.835 | 0.296 |
| Random Forest | 0.659 | 0.873 | 0.831 | 0.346 |
| Trees combined | 0.681 | 0.883 | 0.840 | 0.298 |
| GRU | 0.683 | 0.881 | — | — |
| LSTM | 0.684 | 0.880 | — | — |
| TCN | **0.686** | 0.879 | — | — |

References on the same bench: simple evidence + smoother **0.343**; organisers’ majority vote **0.208**.

**Best tuned configs**

- LightGBM: `num_leaves=31`, `min_child_samples=50`, `colsample_bytree=0.3`, `subsample=0.7`, `reg_lambda=1.0`
- XGBoost: `max_depth=4`, `min_child_weight=1`, `colsample_bytree=0.3`, `subsample=0.7`, `reg_lambda=1.0`
- RF: `max_features=0.2`, `min_samples_leaf=20`

**Verdict:** strong on degraded IDSSE (~2× the weak reference on spellF1).

## 2. BAR-ATH public 5 minutes (smoke test)

Best IDSSE models were retrained on all 7 matches and applied to BAR-ATH ETL features. Public GT was **not** used for training or cost selection.

| Rank | Submission | macroF1 | spellF1 | Pred possessions (gt=17) |
|---:|---|---:|---:|---:|
| 1 | idsse_rf | **0.834** | **0.496** | 8 |
| 2 | idsse_lgbm | 0.822 | 0.477 | 9 |
| 3 | idsse_combined | 0.808 | 0.477 | 9 |
| 4 | idsse_xgb | 0.795 | 0.459 | 10 |
| 5 | baseline_smooth125 | 0.581 | 0.424 | 16 |
| 6 | baseline_smooth50 | 0.621 | 0.326 | 26 |

Public GT home share of live play in the window: **57.3%**.

**Verdict:** better than all organiser baselines (especially macroF1). spellF1 gain is real but modest; the decoder merges possessions (too few, too long). With only 17 GT possessions, scores are noisy.

## 3. Full-match label-free check (graded frames)

| Model | Ball in play | Stoppages | Possessions | Mean spell |
|---|---:|---:|---:|---:|
| idsse_rf | 59.5 min | 110 | 171 | 20.9 s |
| idsse_lgbm | 59.6 min | 109 | 193 | 18.5 s |
| idsse_combined | 59.5 min | 107 | 179 | 19.9 s |
| idsse_xgb | 60.0 min | 111 | 188 | 19.1 s |
| baseline_smooth125 | 82.3 min | 79 | 398 | 12.4 s |

Football priors: ball in play ~52–58 min; stoppages ~100–130; mean possession ~14 s. Stoppage count looks right; ball-in-play a bit high; spells a bit long (same merge issue).

Not used, on purpose: published possession statistics for BAR-ATH. RULES.md forbids any human-produced
data about the provided game ("possession data or annotations of any kind"), also as a sanity check.
An earlier version of this table compared the models' home share with such a figure; it was removed
(2026-11-16).

## 3b. Runs on 2026-11-16 (Windows, RTX 3060) and the final configuration

Smoother calibration without BAR-ATH labels (`src/idsse/rate_match.py`, `src/idsse/barath.py`): on the
IDSSE folds the best fixed smoother costs decode ~91% of the true possessions, and matching the true
count exactly scores slightly worse (0.667 vs 0.681). On BAR-ATH the IDSSE-picked cost merged
possessions (3.30 per live minute), so the team-switch cost is set on BAR-ATH to reproduce the IDSSE
behaviour (3.75 per live minute). The graded frames before the 2nd-half kickoff are forced dead.

| Run (BAR-ATH public 5 min, smoke test only) | spellF1 | macroF1 | possessions (gt 17) |
|---|---:|---:|---:|
| LightGBM, fixed smoother | 0.629 | 0.829 | 11 |
| LightGBM, calibrated smoother | 0.680 | 0.827 | 13 |
| LightGBM 0.5 + TCN 0.25 + GRU 0.25, calibrated (final configuration) | 0.613 | 0.855 | 13 |

The final configuration was chosen on the IDSSE folds (spellF1 0.690, macroF1 0.888), not on these rows.

Experiments that did not help (IDSSE folds, spellF1, all settings chosen per fold on training matches):

| Experiment | Result |
|---|---|
| Tree + network mixes (15 weightings) | 0.677-0.694, no mix clearly better |
| Snap possession changes to the new team's first touch | 0.690 -> 0.690 |
| Turnover model + frame-varying switch cost | 0.689 -> 0.693 |
| Dead-ball detector (src/model/deadball.py) as extra D evidence | every fold picked "no fusion" |
| Two-stage decoding (dead/live, then team per live stretch) | 0.676 (worse) |
| Remove short possessions after decoding | 0.687 |
| Touch-chain inputs (last / next three touches, chains) | LightGBM 0.674, GRU 0.678 (worse) |
| Labels with turnovers snapped to the first touch | 0.677; DFL's flag already changes within ~0.1 s of it |

Diagnostics: making the team right on live frames with the ball visible would lift the bench to 0.90,
with the ball unseen only to 0.75, at restarts to 0.71. A model trained and tested on clean IDSSE reaches
0.77, one trained on clean and tested on degraded data 0.48: training on degraded data is essential, and
the remaining gap is mostly ambiguous turnovers, not tracking noise.

## 4. How to reproduce

To build the submission from the committed models: **docs/IDSSE_SUBMISSION.md**
(`python -m src.idsse.final predict`). The commands below are the full retraining run on a Mac.

```bash
export BARATH_CACHE=/path/to/etl/cache   # needs frames.parquet
# tables + tune + final + NNs (Apple Silicon: MPS / CPU, no CUDA required)
./scripts/train_all.sh
# BAR-ATH public scoreboard submissions
python3 scripts/predict_barath_public.py
python3 evaluation/run_evaluation.py
```

Artifacts (gitignored, local only): `cache/idsse/tuning_*.csv`, `cv_trees_final.csv`, `cvpred_*.npz`, `evaluation/results/idsse_*.csv`.

## 5. Takeaways

1. IDSSE bench numbers are **good**.
2. Transfer to BAR-ATH public window is **positive** vs baselines, not finished.
3. Main gap: **under-segmentation** (merged possessions) — tune decoder costs / duration prior next, not more tree capacity.
   This must be done without BAR-ATH labels: never on the public window (EVALUATION.md, RULES.md).
4. The public-window ranking of the four submissions (RF first) is not a basis for choosing a model:
   17 possessions cannot rank versions. Models are chosen on the IDSSE folds only.
