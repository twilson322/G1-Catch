"""Tuning of the catch parameters on a fixed development set.

The two parameters that decide whether a ball that reaches the cup stays in it (how fast the cup
retreats, and how quickly it brakes afterwards) are chosen here, on development seeds 0 and 1.
No reported result uses those seeds: the experiments in this repository use seeds 500 and above.

    python scripts/tune.py
"""
import itertools

import numpy as np

from g1catch.experiment import RESULTS, Trial, catch_summary, run_trials, write_rows
from g1catch.runtime import RuntimeConfig
from g1catch.sim import ThrowDistribution

DEV_SEEDS = (0, 1)
THROWS_PER_SEED = 50
VELOCITY_MATCH = (0.25, 0.35, 0.45, 0.55)
FOLLOW_THROUGH_S = (0.03, 0.06, 0.12)


def main():
    dist = ThrowDistribution()
    trials = []
    for match, follow in itertools.product(VELOCITY_MATCH, FOLLOW_THROUGH_S):
        for seed in DEV_SEEDS:
            cfg = RuntimeConfig()
            cfg.intercept.velocity_match = match
            cfg.follow_through_s = follow
            trials.append(Trial(f"{match}|{follow}|{seed}", cfg, dist, THROWS_PER_SEED, seed))
    out = run_trials(trials)
    table = []
    for match, follow in itertools.product(VELOCITY_MATCH, FOLLOW_THROUGH_S):
        rows = [r for seed in DEV_SEEDS for r in out[f"{match}|{follow}|{seed}"]]
        s = catch_summary(rows)
        table.append({"velocity_match": match, "follow_through_s": follow, "caught": s["caught"], "n": s["n"],
                      "rate": s["rate"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                      "median_rel_speed": float(np.nanmedian([r["rel_speed"] for r in rows]))})
        print(f"velocity match {match:.2f}, follow-through {follow:.2f} s: {s['caught']}/{s['n']} = {100 * s['rate']:.0f} % "
              f"[{100 * s['ci_lo']:.0f}, {100 * s['ci_hi']:.0f}]", flush=True)
    write_rows(RESULTS / "tuning.csv", table)
    best = max(table, key=lambda r: r["rate"])
    default = RuntimeConfig()
    print(f"best on the development set: velocity match {best['velocity_match']}, follow-through {best['follow_through_s']} s. "
          f"Defaults in the code: {default.intercept.velocity_match}, {default.follow_through_s} s.")


if __name__ == "__main__":
    main()
