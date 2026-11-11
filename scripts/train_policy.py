"""Distil the reach MPC into a neural network with DAgger.

Round 0 is behaviour cloning: the MPC drives and its actions are recorded. In later rounds the
network drives, and every state it visits is labelled with what the MPC would have done there.
After each round the network is evaluated on the same held-out throws.

    python scripts/train_policy.py
"""
import csv
import time

import numpy as np
import torch

from g1catch.experiment import (DETECTOR_MISS_RATE, DETECTOR_PIXEL_SIGMA, RESULTS, Trial, catch_summary,
                                run_trials)
from g1catch.mpc import ReachMpc
from g1catch.policy import DaggerPlanner, Dataset, JointPolicyNet, PolicyPlanner, fit
from g1catch.runtime import CatchRunner, RuntimeConfig
from g1catch.sensing import SyntheticSensor
from g1catch.sim import ARM_DOF, CatchSim, ThrowDistribution
from g1catch.vision import Intrinsics

ROUNDS = 4
THROWS_PER_ROUND = 100
EVAL_THROWS = 120
EVAL_SEED = 3900
EPOCHS = 40


def collect(net, dataset: Dataset, throws: int, seed: int) -> None:
    """Run throws with the network driving (or the MPC if net is None) and label every plan."""
    sim = CatchSim()
    intr = Intrinsics.from_fovy(sim.model.cam_fovy[0])
    cfg = RuntimeConfig()
    expert = ReachMpc(sim.joint_lo, sim.joint_hi, cfg.mpc)
    learner = None if net is None else PolicyPlanner(net, sim.joint_lo, sim.joint_hi, cfg.mpc)
    sensor = SyntheticSensor(intr, np.random.default_rng(seed + 1), DETECTOR_PIXEL_SIGMA, DETECTOR_MISS_RATE)
    runner = CatchRunner(sim, sensor, cfg, reach_planner=DaggerPlanner(expert, learner, dataset))
    rng = np.random.default_rng(seed)
    dist = ThrowDistribution()
    for _ in range(throws):
        runner.run(dist.sample(rng))


def timing_and_limits(net, seed: int = 0, samples: int = 300):
    """Wall time per plan and how often a plan leaves the joint range, on random reach problems."""
    sim = CatchSim()
    cfg = RuntimeConfig().mpc
    mpc = ReachMpc(sim.joint_lo, sim.joint_hi, cfg)
    policy = PolicyPlanner(net, sim.joint_lo, sim.joint_hi, cfg)
    rng = np.random.default_rng(seed)
    out = {}
    unfiltered = PolicyPlanner(net, sim.joint_lo, sim.joint_hi, cfg, safety_filter=False)
    for name, planner in (("mpc", mpc), ("policy, no filter", unfiltered), ("policy + barrier filter", policy)):
        rng = np.random.default_rng(seed)
        wall, violations, terminal = [], 0, []
        for _ in range(samples):
            q0 = rng.uniform(mpc.q_lo, mpc.q_hi)
            dq0 = rng.normal(0, 1.0, ARM_DOF)
            q_goal = np.clip(q0 + rng.normal(0, 0.4, ARM_DOF), mpc.q_lo, mpc.q_hi)
            dq_goal = rng.normal(0, 1.5, ARM_DOF)
            horizon = rng.uniform(0.08, 0.4)
            t0 = time.perf_counter()
            sol = planner.solve(q0, dq0, q_goal, dq_goal, horizon)
            wall.append(time.perf_counter() - t0)
            q, dq = q0.copy(), dq0.copy()
            left_range = False
            for u in sol.accels:
                q = q + sol.step * dq + 0.5 * sol.step**2 * u
                dq = dq + sol.step * u
                left_range |= bool(np.any(q < mpc.q_lo - 1e-3) or np.any(q > mpc.q_hi + 1e-3))
            violations += left_range
            terminal.append(np.abs(q - q_goal).max())
        out[name] = {"plan_ms": 1e3 * float(np.median(wall)), "plans_leaving_joint_range": violations / samples,
                     "median_terminal_error_rad": float(np.median(terminal))}
    return out


def main():
    torch.manual_seed(0)
    net = JointPolicyNet()
    dataset = Dataset([], [])
    dist = ThrowDistribution()
    log = []
    baseline = run_trials([Trial("mpc", RuntimeConfig(), dist, EVAL_THROWS, EVAL_SEED)])["mpc"]
    s = catch_summary(baseline)
    print(f"MPC on the evaluation throws: {s['caught']}/{s['n']} = {100 * s['rate']:.1f} %", flush=True)
    log.append({"round": "mpc", "samples": 0, "train_loss": float("nan"), "caught": s["caught"], "n": s["n"],
                "rate": s["rate"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "median_miss_cm": s["median_miss_cm"]})
    for rnd in range(ROUNDS):
        t0 = time.time()
        collect(None if rnd == 0 else net, dataset, THROWS_PER_ROUND, seed=3000 + rnd)
        loss = fit(net, dataset, epochs=EPOCHS, seed=rnd)
        state = {k: v.clone() for k, v in net.state_dict().items()}
        rows = run_trials([Trial(f"round {rnd}", RuntimeConfig(), dist, EVAL_THROWS, EVAL_SEED, policy_state=state)])[f"round {rnd}"]
        s = catch_summary(rows)
        n_samples = len(dataset.arrays()[0])
        label = "behaviour cloning" if rnd == 0 else f"DAgger round {rnd}"
        print(f"{label:18s} {n_samples:6d} samples, loss {loss:.4f}, caught {s['caught']}/{s['n']} = {100 * s['rate']:.1f} % "
              f"[{100 * s['ci_lo']:.0f}, {100 * s['ci_hi']:.0f}], miss {s['median_miss_cm']:.1f} cm ({time.time() - t0:.0f} s)", flush=True)
        log.append({"round": label, "samples": n_samples, "train_loss": loss, "caught": s["caught"], "n": s["n"],
                    "rate": s["rate"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "median_miss_cm": s["median_miss_cm"]})
    raw = run_trials([Trial("no filter", RuntimeConfig(), dist, EVAL_THROWS, EVAL_SEED, policy_state=state,
                            policy_safety_filter=False)])["no filter"]
    s = catch_summary(raw)
    print(f"final network without the barrier filter: caught {s['caught']}/{s['n']} = {100 * s['rate']:.1f} %", flush=True)
    log.append({"round": "final, no barrier filter", "samples": n_samples, "train_loss": loss, "caught": s["caught"],
                "n": s["n"], "rate": s["rate"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "median_miss_cm": s["median_miss_cm"]})
    torch.save(net.state_dict(), RESULTS / "reach_policy.pt")
    with open(RESULTS / "dagger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    compare = timing_and_limits(net)
    with open(RESULTS / "policy_vs_mpc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["planner", "plan_ms", "plans_leaving_joint_range", "median_terminal_error_rad"])
        for name, c in compare.items():
            w.writerow([name, c["plan_ms"], c["plans_leaving_joint_range"], c["median_terminal_error_rad"]])
            print(name, {k: round(v, 4) for k, v in c.items()})


if __name__ == "__main__":
    main()
