# Catalysts — More than a Hack 2027, Phase 1: Ball Possession

Team **Catalysts**. One label (`H`, `A`, `D` or `X`) for every frame of the Barcelona vs Athletic
Club tactical video.

## Reproduce `Catalysts.csv` (≈ 5 minutes, CPU)

```bash
# 1. unpack the organisers' match bundle into data/ (see docs/challenge/DATA.md)
pip install -r requirements-etl.txt -r requirements-idsse.txt

# 2. build the frame table from the tactical ATD
python -m src.etl.build                         # writes cache/frames.parquet

# 3. apply the committed models
python -m src.idsse.final predict               # writes Catalysts.csv + prints sha256

# 4. format check (same checks the official scorer runs)
python evaluation/validate_submission.py Catalysts.csv
```

`predict` reads only `models/idsse/` (in this repository) and `cache/frames.parquet`. Point
`BARATH_CACHE` at an existing ETL cache if you already have one. Networks run on CPU in float32
with one thread so the same inputs produce the same file; run twice and compare the sha256.

Optional smoke test on the public five minutes (never used to choose the model):

```bash
cp Catalysts.csv evaluation/results/
python evaluation/run_evaluation.py --gt public
```

Full write-up of the build steps: [docs/IDSSE_SUBMISSION.md](docs/IDSSE_SUBMISSION.md).  
Results and experiment log: [docs/IDSSE_RESULTS.md](docs/IDSSE_RESULTS.md).  
Technical document for the organisers: [Catalysts.md](Catalysts.md) / [Catalysts.pdf](Catalysts.pdf).

## What the submission is

- **ETL (etl-v1.3):** deterministic cleaning and frame features from the tactical ATD (measured
  positions only near the ball, keeper-slot correction, contact needs ball reaction, etc.).
- **Models:** LightGBM (0.5) + TCN (0.25) + GRU (0.25), each averaged with its home/away mirror.
  Trained on **IDSSE** (7 DFL Bundesliga matches, CC BY 4.0), degraded to look like automatic
  tracking. No labels from the provided BAR-ATH game.
- **Decoder:** Viterbi over H/A/D. Dead-ball cost from IDSSE folds; team-switch cost calibrated on
  BAR-ATH so the possession *rate* matches IDSSE behaviour (**no BAR-ATH labels**). Graded frames
  before the 2nd-half kickoff are dead (FAQ). Ungraded frames are `X`.

## Data that must not be in the repo

The match bundle stays out of git (`data/`, `cache/`, `Catalysts.csv` are gitignored). IDSSE raw
files go under `external-data/idsse/` (also gitignored); download with
`python -c "from src.idsse import download; download.ensure()"`.

## Licence

See [LICENSE](LICENSE). IDSSE: Bassek et al., Scientific Data 2025, figshare DOI 10.6084/m9.figshare.28196177 (CC BY 4.0).
