"""How long the planner takes per camera frame, measured on an otherwise idle machine.

The runtime assumes a plan takes effect 10 ms after its frame arrives (RuntimeConfig.planning_delay_s).
This script measures the wall time of every planner call over a batch of throws in one process and
compares it with that budget, and times the compiled flight model against its NumPy reference.

    python scripts/planner_timing.py
"""
import csv
import time

import numpy as np

from g1catch import ballistics as bal
from g1catch import ukf
from g1catch.experiment import DETECTOR_MISS_RATE, DETECTOR_PIXEL_SIGMA, RESULTS
from g1catch.runtime import CatchRunner, RuntimeConfig
from g1catch.sensing import SyntheticSensor
from g1catch.sim import CatchSim, ThrowDistribution
from g1catch.vision import Intrinsics

THROWS = 80
SEED = 8000


def main():
    sim = CatchSim()
    cfg = RuntimeConfig()
    intr = Intrinsics.from_fovy(sim.model.cam_fovy[0])
    runner = CatchRunner(sim, SyntheticSensor(intr, np.random.default_rng(SEED + 1), DETECTOR_PIXEL_SIGMA, DETECTOR_MISS_RATE), cfg)
    rng = np.random.default_rng(SEED)
    dist = ThrowDistribution()
    first, later = [], []
    for _ in range(THROWS):
        result = runner.run(dist.sample(rng))
        if result.planner_ms:
            # The slowest call of a throw is the one that chooses the catch point: a one-second
            # forecast and up to four full inverse-kinematics solves.
            slowest = int(np.argmax(result.planner_ms))
            first.append(result.planner_ms[slowest])
            later += [t for i, t in enumerate(result.planner_ms) if i != slowest]
    budget = 1e3 * cfg.planning_delay_s
    rows = []
    for name, values in (("choosing the catch point (once per throw)", first), ("re-planning (every other frame)", later),
                         ("all calls", first + later)):
        v = np.array(values)
        rows.append({"call": name, "count": len(v), "median_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90)),
                     "p99_ms": float(np.percentile(v, 99)), "max_ms": float(v.max()),
                     "over_budget": float(np.mean(v > budget))})
        print(f"{name:44s} n {len(v):5d}  median {np.median(v):5.2f} ms  p90 {np.percentile(v, 90):5.2f}  "
              f"p99 {np.percentile(v, 99):5.2f}  max {v.max():5.2f}  over the {budget:.0f} ms budget {100 * np.mean(v > budget):.1f} %")
    with open(RESULTS / "planner_timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    states = np.tile([3.0, 0.0, 1.2, -5.0, 0.2, 2.5], (13, 1)) + np.random.default_rng(0).normal(0, 0.05, (13, 6))
    timings = {}
    for name, fn in (("C++", lambda: ukf.rollout(states, 0.005, 200, bal.NOMINAL_KD)),
                     ("NumPy", lambda: [ukf.propagate_reference(states, 0.005, bal.NOMINAL_KD) for _ in range(200)])):
        fn()
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        timings[name] = 1e3 * (time.perf_counter() - t0) / 20
    print(f"one-second forecast of 13 sigma points: C++ {timings['C++']:.3f} ms, NumPy {timings['NumPy']:.1f} ms "
          f"({timings['NumPy'] / timings['C++']:.0f}x)")
    with open(RESULTS / "flight_core_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["implementation", "forecast_ms"])
        w.writerows(timings.items())


if __name__ == "__main__":
    main()
