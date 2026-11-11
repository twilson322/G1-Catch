"""Catch-rate experiments with the pixel-noise sensor: baseline, ablations and sweeps.

Every condition in a group runs the same seeded throws, so conditions are compared throw by
throw (exact McNemar test on the throws where the two outcomes differ).

    python scripts/run_experiments.py            # everything, about an hour on two cores
    python scripts/run_experiments.py ablations  # one group
"""
import csv
import sys

import numpy as np
from scipy.stats import binomtest

from g1catch.experiment import RESULTS, Trial, catch_summary, run_trials, write_rows
from g1catch.runtime import RuntimeConfig
from g1catch.sim import ThrowDistribution

BASELINE_THROWS = 300        # split over two seeds so both cores work
GROUP_THROWS = 120
SEED = 1000


def variant(**changes) -> RuntimeConfig:
    """A default config with dotted-path overrides, e.g. variant(**{"intercept.velocity_match": 0.0})."""
    cfg = RuntimeConfig()
    for path, value in changes.items():
        target = cfg
        *parents, leaf = path.split(".")
        for name in parents:
            target = getattr(target, name)
        if not hasattr(target, leaf):
            raise AttributeError(f"no config field {path}")
        setattr(target, leaf, value)
    return cfg


def groups() -> dict[str, list[Trial]]:
    direct = ThrowDistribution()
    bounce = ThrowDistribution(bounce=True)
    n = GROUP_THROWS
    g = {}
    g["baseline"] = [Trial("baseline, seed a", RuntimeConfig(), direct, BASELINE_THROWS // 2, SEED),
                     Trial("baseline, seed b", RuntimeConfig(), direct, BASELINE_THROWS // 2, SEED + 50)]
    g["ablations"] = [
        Trial("full system", RuntimeConfig(), direct, n, SEED + 1),
        Trial("no velocity matching", variant(**{"intercept.velocity_match": 0.0}), direct, n, SEED + 1),
        Trial("plan once at commit", variant(replan_every_frame=False), direct, n, SEED + 1),
        Trial("no latency compensation", variant(compensate_latency=False), direct, n, SEED + 1),
        Trial("no impedance at contact", variant(use_impedance=False), direct, n, SEED + 1),
        Trial("slow follow-through (0.15 s)", variant(follow_through_s=0.15), direct, n, SEED + 1),
        Trial("single filter, no IMM", variant(**{"tracker.use_imm": False}), direct, n, SEED + 1),
        Trial("no drag in the flight model", RuntimeConfig(), direct, n, SEED + 1, kd=0.0),
    ]
    g["realism"] = [
        Trial("default: 15-bit encoders, Kalman velocity, 10 ms planning delay", RuntimeConfig(), direct, n, SEED + 7),
        Trial("true joint state", variant(**{"controller.sensing.observer": "exact"}), direct, n, SEED + 7),
        Trial("differenced velocity", variant(**{"controller.sensing.observer": "difference"}), direct, n, SEED + 7),
        Trial("12-bit encoders", variant(**{"controller.sensing.encoder_bits": 12}), direct, n, SEED + 7),
        Trial("no planning delay", variant(planning_delay_s=0.0), direct, n, SEED + 7),
        Trial("20 ms planning delay", variant(planning_delay_s=0.020), direct, n, SEED + 7),
        Trial("35 ms planning delay", variant(planning_delay_s=0.035), direct, n, SEED + 7),
        Trial("no self-collision check", variant(check_self_collision=False), direct, n, SEED + 7),
    ]
    g["latency"] = [Trial(f"{ms} ms", variant(perception_latency_s=ms / 1000), direct, n, SEED + 2)
                    for ms in (10, 30, 50, 70, 100)]
    g["latency"] += [Trial(f"{ms} ms uncompensated", variant(perception_latency_s=ms / 1000, compensate_latency=False),
                           direct, n, SEED + 2) for ms in (10, 30, 50)]
    g["pixel_noise"] = [Trial(f"{s} px", variant(**{"tracker.pixel_sigma": s}), direct, n, SEED + 3, pixel_sigma=s)
                        for s in (0.25, 0.5, 1.0, 2.0)]
    g["restitution"] = [Trial(f"e = {e}", RuntimeConfig(), direct, n, SEED + 4, cup_restitution=e)
                        for e in (0.1, 0.2, 0.35, 0.5)]
    g["restitution"] += [Trial(f"e = {e}, no velocity matching", variant(**{"intercept.velocity_match": 0.0}), direct, n,
                               SEED + 4, cup_restitution=e) for e in (0.1, 0.2)]
    g["bounce"] = [Trial("bounce, IMM", RuntimeConfig(), bounce, n, SEED + 5),
                   Trial("bounce, single filter", variant(**{"tracker.use_imm": False}), bounce, n, SEED + 5),
                   Trial("bounce, cup aimed straight against the ball",
                         variant(**{"intercept.min_axis_up": -1.0, "intercept.rising_penalty_s": 0.0}), bounce, n, SEED + 5)]
    g["outliers"] = [Trial("5 % false detections, gated", RuntimeConfig(), direct, n, SEED + 6, outlier_rate=0.05),
                     Trial("5 % false detections, gate off", variant(**{"tracker.gate_probability": 1.0}), direct, n,
                           SEED + 6, outlier_rate=0.05)]
    return g


def mcnemar(reference: list[dict], other: list[dict]) -> float:
    """Exact two-sided p-value that the two conditions catch the same fraction of the same throws."""
    only_ref = sum(a["caught"] and not b["caught"] for a, b in zip(reference, other))
    only_other = sum(b["caught"] and not a["caught"] for a, b in zip(reference, other))
    if only_ref + only_other == 0:
        return 1.0
    return float(binomtest(only_ref, only_ref + only_other, 0.5).pvalue)


def summarise(group: str, trials: list[Trial], out: dict[str, list[dict]]) -> list[dict]:
    reference = out[trials[0].name]
    summary = []
    print(f"== {group}", flush=True)
    for t in trials:
        r = out[t.name]
        s = catch_summary(r)
        paired = t is not trials[0] and len(r) == len(reference) and t.seed == trials[0].seed
        s.update({"group": group, "trial": t.name,
                  "committed": float(np.mean([x["committed"] for x in r])),
                  "median_rel_speed": float(np.nanmedian([x["rel_speed"] for x in r])),
                  "median_forecast_error_cm": float(np.nanmedian([x["forecast_error_cm"] for x in r])),
                  "median_cup_error_cm": float(np.nanmedian([x["cup_error_cm"] for x in r])),
                  "planner_ms_mean": float(np.nanmean([x["planner_ms"] for x in r])),
                  "p_vs_first": mcnemar(reference, r) if paired else float("nan")})
        summary.append(s)
        print(f"  {t.name:34s} {s['caught']:3d}/{s['n']} = {100 * s['rate']:5.1f} % [{100 * s['ci_lo']:.0f}, {100 * s['ci_hi']:.0f}]"
              f"  miss {s['median_miss_cm']:.1f} cm  forecast {s['median_forecast_error_cm']:.1f} cm"
              f"  rel speed {s['median_rel_speed']:.1f} m/s  p {s['p_vs_first']:.3g}", flush=True)
    return summary


def load_group(group: str, trials: list[Trial]) -> dict[str, list[dict]]:
    """Throw rows of a finished group, read back from its CSV file."""
    numeric = ("caught", "committed", "rel_speed", "forecast_error_cm", "cup_error_cm", "planner_ms", "miss_cm",
               "self_collided")
    out = {t.name: [] for t in trials}
    with open(RESULTS / f"exp_{group}.csv") as f:
        for row in csv.DictReader(f):
            for key in numeric:
                row[key] = float(row[key]) if row[key] not in ("", "nan") else float("nan")
            row["caught"], row["committed"] = int(row["caught"]), int(row["committed"])
            out[row["trial"]].append(row)
    return out


def main():
    """With group names: run those groups. Every group's summary is rebuilt from the CSV files on
    disk, so a group that was run earlier keeps its place in exp_summary.csv."""
    all_groups = groups()
    wanted = sys.argv[1:] or list(all_groups)
    for group in wanted:
        trials = all_groups[group]
        out = run_trials(trials)
        write_rows(RESULTS / f"exp_{group}.csv", [row for t in trials for row in out[t.name]])
    summary = []
    for group, trials in all_groups.items():
        if (RESULTS / f"exp_{group}.csv").exists():
            summary += summarise(group, trials, load_group(group, trials))
    write_rows(RESULTS / "exp_summary.csv", summary)


if __name__ == "__main__":
    main()
