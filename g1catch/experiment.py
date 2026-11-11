"""Batch evaluation helpers shared by the scripts: seeded throws, parallel workers, CSV output."""
from __future__ import annotations

import csv
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .runtime import CatchRunner, RuntimeConfig, ThrowResult
from .sensing import SyntheticSensor
from .sim import CatchSim, ThrowDistribution
from .vision import Intrinsics

RESULTS = Path(__file__).resolve().parents[1] / "results"
# Stand-in for the detector when images are not rendered. Set from results/detector_eval.csv.
DETECTOR_PIXEL_SIGMA = 0.5
DETECTOR_MISS_RATE = 0.02


@dataclass
class Trial:
    """One batch of throws. Must be picklable: it crosses a process boundary."""
    name: str
    cfg: RuntimeConfig
    dist: ThrowDistribution
    throws: int
    seed: int
    pixel_sigma: float = DETECTOR_PIXEL_SIGMA
    miss_rate: float = DETECTOR_MISS_RATE
    outlier_rate: float = 0.0
    cup_restitution: float | None = None
    residual: object = None
    kd: float | None = None
    keep_frames: bool = False
    sensor: str = "synthetic"         # "synthetic", "cnn" or "blob" (the last two render images)
    appearance: str = "nominal"       # "nominal", "randomized" (a new look per throw) or "hard"
    catch_gate: object = None         # callable(features) -> bool, see g1catch/catchability.py
    intercept_adjust: object = None   # callable(features) -> (dt, dp), see g1catch/ppo.py
    policy_state: dict | None = None  # JointPolicyNet weights: replaces the reach MPC
    policy_safety_filter: bool = True # barrier-function filter behind the network


def summarise(result: ThrowResult, throw, index: int) -> dict:
    row = {
        "throw": index, "caught": int(result.caught), "committed": int(result.committed),
        "touched": int(result.touched), "self_collided": int(result.self_collided), "miss_cm": 100 * result.miss_distance_m,
        "forecast_error_cm": 100 * result.forecast_error_m, "cup_error_cm": 100 * result.cup_error_m,
        "reference_error_cm": 100 * result.reference_error_m,
        "aim_error_deg": float(np.degrees(result.aim_error_rad)),
        "flight_time": throw.flight_time, "commit_time": result.commit_time, "catch_time": result.catch_time,
        "rel_speed": result.relative_speed_at_contact, "peak_torque": result.peak_torque_fraction,
        "saturated_ticks": result.saturated_ticks, "saturated_ticks_reach": result.saturated_ticks_reach,
        "saturated_joint": result.saturated_joint, "reach_tracking_mrad": 1e3 * result.reach_tracking_error_rad, "planner_over_budget": result.planner_over_budget,
        "planner_calls": len(result.planner_ms),
        "planner_ms": float(np.mean(result.planner_ms)) if result.planner_ms else float("nan"),
        "planner_ms_max": float(np.max(result.planner_ms)) if result.planner_ms else float("nan"),
        "drag_coefficient": throw.drag_coefficient, "spin": float(np.linalg.norm(throw.spin)),
    }
    if result.commit_features:
        row.update({f"f_{k}": v for k, v in result.commit_features.items()})
    return row


def _appearance(kind: str, rng: np.random.Generator):
    from .render import Appearance
    if kind == "nominal":
        return Appearance.nominal()
    if kind == "randomized":
        return Appearance.random(rng, 1.0)
    look = Appearance.random(rng, 1.5)
    look.distractor = True
    return look


def run_trial(trial: Trial) -> list[dict]:
    """Run one batch. Throws depend only on trial.seed, so two trials with the same seed and
    distribution see the same throws whatever else differs."""
    sim = CatchSim()
    if trial.cup_restitution is not None:
        sim.set_cup_restitution(trial.cup_restitution)
    intr = Intrinsics.from_fovy(sim.model.cam_fovy[0])
    throw_rng = np.random.default_rng(trial.seed)
    sensor_rng = np.random.default_rng(trial.seed + 1_000_003)
    if trial.sensor == "synthetic":
        sensor = SyntheticSensor(intr, sensor_rng, trial.pixel_sigma, trial.miss_rate, trial.outlier_rate)
    else:
        # Imported here so that experiments without images do not need an OpenGL context.
        from .detector import CnnDetector
        from .render import StereoRenderer
        from .sensing import RenderedSensor
        from .vision import BlobDetector
        detector = CnnDetector() if trial.sensor == "cnn" else BlobDetector()
        sensor = RenderedSensor(intr, StereoRenderer(sim.model), detector, sensor_rng)
    look_rng = np.random.default_rng(trial.seed + 2_000_003)
    planner = None
    if trial.policy_state is not None:
        from .policy import JointPolicyNet, PolicyPlanner
        net = JointPolicyNet()
        net.load_state_dict(trial.policy_state)
        planner = PolicyPlanner(net, sim.joint_lo, sim.joint_hi, trial.cfg.mpc, safety_filter=trial.policy_safety_filter)
    runner = CatchRunner(sim, sensor, trial.cfg, residual=trial.residual, kd=trial.kd, reach_planner=planner,
                         catch_gate=trial.catch_gate, intercept_adjust=trial.intercept_adjust)
    rows = []
    for i in range(trial.throws):
        throw = trial.dist.sample(throw_rng)
        if trial.sensor != "synthetic":
            sensor.look = _appearance(trial.appearance, look_rng)
        result = runner.run(throw)
        row = summarise(result, throw, i)
        row["trial"] = trial.name
        if trial.keep_frames:
            row["_frames"] = result.frames
        decision = getattr(trial.intercept_adjust, "last", None)
        if decision is not None:
            row["_decision"] = decision                 # (observation, action, log-probability)
            trial.intercept_adjust.last = None
        rows.append(row)
    return rows


def run_trials(trials: list[Trial], workers: int = 2) -> dict[str, list[dict]]:
    if workers <= 1 or len(trials) == 1:
        out = [run_trial(t) for t in trials]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            out = pool.map(run_trial, trials, chunksize=1)
    return {t.name: rows for t, rows in zip(trials, out)}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def write_rows(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        for k in r:
            if not k.startswith("_") and k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def catch_summary(rows: list[dict]) -> dict:
    n = len(rows)
    caught = sum(r["caught"] for r in rows)
    lo, hi = wilson_interval(caught, n)
    return {"n": n, "caught": caught, "rate": caught / n, "ci_lo": lo, "ci_hi": hi,
            "median_miss_cm": float(np.median([r["miss_cm"] for r in rows])),
            "within_3cm": float(np.mean([r["miss_cm"] < 3.0 for r in rows])),
            "self_collided": float(np.mean([r["self_collided"] for r in rows]))}
