# Building the submission from the IDSSE-trained models

Everything needed is in this branch. The match data is not (it never goes into the repository).

## Build Catalysts.csv (about 5 minutes, CPU only)

```bash
pip install -r requirements-etl.txt -r requirements-idsse.txt
# 1. unpack the match bundle into data/ (DATA.md), then build the team's ETL cache:
python -m src.etl.build                         # writes cache/frames.parquet (+ report)
# 2. apply the saved models to BAR-ATH:
python -m src.idsse.final predict               # writes Catalysts.csv and prints its sha256
# 3. check it exactly as the scorer will:
python evaluation/validate_submission.py Catalysts.csv
```

`predict` reads only `models/idsse/` (committed) and `cache/frames.parquet`. If the ETL cache lives
elsewhere, point to it with `BARATH_CACHE=/path/to/cache`. The networks run on the CPU in float32 with
one thread, so the same inputs give the same file on any machine: run it twice and compare the sha256.

Optional smoke check on the public five minutes (never used to choose anything):
`python evaluation/run_evaluation.py --gt public` after copying the file to `evaluation/results/`.

## What the file is

- **Inputs:** the etl-v1.3 frames table plus 5 s and 15 s context (`src/idsse/prep.py`); no Smart
  Tagging, no camera columns, no ball-stillness flags.
- **Models:** LightGBM (0.5) + TCN (0.25) + GRU (0.25), each averaged over the frame and its home/away
  mirror. Trained on 7 IDSSE Bundesliga matches (DFL, CC BY 4.0), degraded to look like BAR-ATH's
  automatic tracking, labels from DFL's possession flag where the DFL event log confirms control.
- **Smoother:** Viterbi over H / A / D. Dead-ball cost from the IDSSE folds; the team-switch cost is
  set on BAR-ATH so the possession rate matches what the IDSSE-picked costs give on IDSSE (no BAR-ATH
  labels). The 2,934 graded frames before the second-half kickoff are dead (FAQ.md). Ungraded frames: X.
- **Scores:** IDSSE cross-validation spellF1 0.690 / macroF1 0.888. Public five minutes (smoke test):
  spellF1 0.61-0.68 / macroF1 0.83-0.86 for this family of runs.

## Retraining (not needed for the submission)

```bash
python -c "from src.idsse import download; download.ensure()"   # IDSSE from figshare (~2.6 GB)
python -m src.model.deadball                                       # dead-ball detector on BAR-ATH (degrader calibration)
python -m src.idsse.build_tables                                   # degraded training tables (~20 min)
python -m src.idsse.rate_match cv                                  # LightGBM fold predictions
python -m src.idsse.train_nn cv tcn && python -m src.idsse.train_nn cv gru
python -m src.idsse.final train                                    # writes models/idsse/
```

Retraining on a GPU is not bit-for-bit repeatable; that is why the trained models are committed and
`predict` only reads them.
