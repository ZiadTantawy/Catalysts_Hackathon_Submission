# Catalysts — Ball Possession Technical Document

**Team:** Catalysts  
**Challenge:** More than a Hack 2027, Phase 1 — Ball Possession (Barcelona vs Athletic Club)  
**Deliverable:** `Catalysts.csv` (199,319 frames)  
**Repository:** https://github.com/ZiadTantawy/Catalysts_Hackathon_Submission

---

## 1. Approach

We treat possession as a **three-state sequence** (`H` / `A` / `D`) over the match, not as
independent frame classification.

1. **ETL (etl-v1.3).** Build a deterministic frame table from the tactical ATD: orientation so home
   attacks +x, measured-only evidence near the ball, keeper-slot team correction, contacts that
   require a ball reaction, and situation flags (crowd, blackout, surplus, etc.). Smart Tagging and
   camera columns exist in the table but are **not** fed to the submitted models.
2. **Per-frame evidence.** A weighted ensemble predicts `P(H,A,D)` on every frame:
   - LightGBM (weight 0.5)
   - temporal CNN / TCN (0.25)
   - bidirectional GRU (0.25)  
   Each head is averaged with its **home↔away mirror** so the model cannot learn a home-club bias.
3. **Temporal decoder.** Viterbi over each graded half with switching costs. A few noisy frames
   cannot open a new possession. Graded frames before the second-half kickoff are forced dead
   (published fact). Ungraded frames are written as `X`.

We chose this over pure nearest-player heuristics because the ranking metric (`spellF1`) rewards
whole possessions, and over training on BAR-ATH labels because the rules forbid that.

---

## 2. Assets used

| Asset | Role |
|---|---|
| Tactical ATD (XML + raw) | Primary input to the ETL and models |
| Homography | Used in ETL camera features; **dropped** from model inputs (domain gap vs IDSSE) |
| Smart Tagging | Used in ETL / dead-ball diagnostics; **dropped** from model inputs (IDSSE has none) |
| Broadcast package | **Not used** |
| Public 5-minute GT | Pipeline smoke test only — **never** for training or model selection |

---

## 3. Training data

**IDSSE** — Bassek et al., *An integrated dataset of spatiotemporal and event data in elite soccer*,
Scientific Data (2025). Figshare article 28196177, DOI 10.6084/m9.figshare.28196177, **CC BY 4.0**.

- **7** DFL Bundesliga matches (clean corrected tracking + ball status/possession + events).
- Labels: `D` when DFL marks the ball dead; otherwise the team from DFL possession where the event
  log confirms control (variant B).
- **Domain adaptation:** each match is deliberately degraded (ball loss, anonymous slots, guessed
  off-screen players, provider-like smoothing) at several severities so the model sees ATD-like
  mess, not only perfect tracking.
- Augmentation: home/away mirror on every training window.
- **No labels from the provided Barcelona–Athletic game** were used for training, validation, or
  hyperparameter choice.

Committed artefacts: `models/idsse/` (LightGBM + TCN + GRU + meta). Retraining on GPU is not
bit-identical; `predict` only loads these files on CPU for a reproducible `Catalysts.csv`.

---

## 4. Possession rules (as implemented)

| Situation | Rule |
|---|---|
| Live play | Team = argmax of smoothed `P(H)` / `P(A)` while not dead |
| Dead ball | `D` when the ensemble + Viterbi prefer dead; includes long stoppages and set-piece holds |
| Pre-2nd-half kickoff (graded) | Forced `D` (2,934 frames; FAQ) |
| Contested / missing ball | Model uses shape, flow, last/next contact context; no hand override |
| Restarts | Possession after a stoppage follows the decoded sequence (no tag-forced team) |
| Ungraded time | `X` (halftime, before/after graded window) |
| Abstention on graded frames | Not used in the final file |

Decoder costs: dead-ball cost from IDSSE leave-one-match-out; **team-switch cost** set on BAR-ATH so
the decoded possession *rate* matches the rate those IDSSE costs produce on IDSSE validation
matches — **without reading BAR-ATH labels**.

---

## 5. Validation and scores

| Check | Result |
|---|---|
| IDSSE CV (model selection) | spellF1 **0.690** / macroF1 **0.888** |
| Public 5 min (smoke only) | spellF1 **0.613** / macroF1 **0.856** (13 pred possessions vs 17 GT) |
| Format | `validate_submission.py` → OK |
| Label-free BAR-ATH priors | ~213 possessions, ~58 min ball-in-play, ~107 stoppages |

Baselines on the public window peak at spellF1 ≈ 0.42. Experiments that did **not** help on IDSSE
(dead-ball fusion, two-stage decode, touch-chain features, etc.) are listed in `docs/IDSSE_RESULTS.md`.

---

## 6. Limitations

- **Turnovers / short spells:** the decoder still merges some possessions (public window: 13 vs 17).
  spellF1 is the main gap versus a perfect labeller.
- **Domain gap:** IDSSE is corrected tracking; BAR-ATH is automatic. Degradation closes much of it
  but not ambiguous loose balls with the ball unseen.
- **No BAR-ATH fine-tuning** by rule — further gains need better label-free rate matching or richer
  ball-free features, not fitting the public five minutes.
- With more time: sharper restart detection, explicit duration priors for short throw-ins, and a
  held-out IDSSE stress set at heavier degradation.

---

## 7. Reproduction

```bash
pip install -r requirements-etl.txt -r requirements-idsse.txt
# unpack match bundle into data/
python -m src.etl.build
python -m src.idsse.final predict          # -> Catalysts.csv
python evaluation/validate_submission.py Catalysts.csv
```

See also `README.md` and `docs/IDSSE_SUBMISSION.md`. Match data and `cache/` are gitignored and must
not be published.
