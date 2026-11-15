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

| Model | Home share | Away share | Ball in play | Stoppages | Possessions | Mean spell |
|---|---:|---:|---:|---:|---:|---:|
| idsse_rf | 63.2% | 36.8% | 59.5 min | 110 | 171 | 20.9 s |
| idsse_lgbm | 62.3% | 37.7% | 59.6 min | 109 | 193 | 18.5 s |
| idsse_combined | 63.5% | 36.5% | 59.5 min | 107 | 179 | 19.9 s |
| idsse_xgb | 63.1% | 36.9% | 60.0 min | 111 | 188 | 19.1 s |
| baseline_smooth125 | 56.7% | 43.3% | 82.3 min | 79 | 398 | 12.4 s |

External match possession figure used for sanity: **Barça 65% / Athletic 35%**. Models ~62–63% / ~37% (~2 pp off).

Football priors: ball in play ~52–58 min; stoppages ~100–130; mean possession ~14 s. Stoppage count looks right; ball-in-play a bit high; spells a bit long (same merge issue).

## 4. How to reproduce

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
4. Full-match possession share matches the published 65/35 split closely enough to pass a sanity check.
