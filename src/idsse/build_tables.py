"""Build every training / test table the IDSSE track needs (cached; re-running skips what exists).

    python -m src.idsse.build_tables            # add --force after changing the degrader

Train/validation: the DEV matches at every mess level, on the training seeds.
Test: the TEST matches at every mess level plus clean, on a seed never used in training.
"""

import sys
import time

from . import dataset, splits


def jobs():
    for seed in splits.TRAIN_SEEDS[:1]:                        # second seed later, if time allows
        for level in dataset.LEVELS:
            for mid in splits.DEV:
                yield mid, level, seed
    for level in list(dataset.LEVELS) + ["clean"]:
        for mid in splits.TEST:
            yield mid, level, splits.TEST_SEED


def main():
    force = "--force" in sys.argv                              # rebuild after a degrader change
    t0 = time.time()
    todo = list(jobs())
    for k, (mid, level, seed) in enumerate(todo, 1):
        df = dataset.match_table(mid, level, seed, force=force)
        print("[{}/{}] {} {} seed {}: {} frames ({:.0f} s)".format(
            k, len(todo), mid, level, seed, len(df), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
