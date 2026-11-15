"""Train / validation / test split of the IDSSE matches. Every later step imports it from here.

Rules:
- Split by match, never by frame: neighbouring frames are near-identical, so a frame-level split leaks.
- Every degraded version of a match (levels, seeds) stays on the same side as the match.
- Test matches are scored once, at the end, on a degradation seed never used in training.

Test: J03WN1 (Bochum - Leverkusen: no Fortuna Düsseldorf, checks the model did not learn one club) and
J03WR9 (Düsseldorf - Kaiserslautern: the typical case). The other five are used for leave-one-match-out
cross-validation: train on four, validate on the fifth, five times; choices are made on the mean.
For the BAR-ATH submission the chosen model is retrained on all seven matches.
"""

TEST = ["J03WN1", "J03WR9"]
DEV = ["J03WMX", "J03WOH", "J03WOY", "J03WPY", "J03WQQ"]
ALL = DEV + TEST

TRAIN_SEEDS = [1, 2]       # degradation seeds for training / validation tables
TEST_SEED = 99             # fresh noise for the test matches, never used in training


def cv_folds():
    """[(train matches, validation match)] for leave-one-match-out over DEV."""
    return [([m for m in DEV if m != v], v) for v in DEV]
