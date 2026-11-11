"""Learned aerodynamic residuals: Gaussian process against boosted trees against physics.

The baseline flight model is gravity only. Calibration throws, observed by a reference tracker
(position at 200 Hz with 2 mm noise, standing in for motion capture), give the residual
acceleration as a function of velocity. Three corrections are compared on held-out throws:
the analytic drag term with a nominal coefficient, a GP, and gradient-boosted trees.

    python scripts/residual_models.py
"""
import csv
import pickle
import time

import numpy as np
from scipy.signal import savgol_filter

from g1catch import ballistics as bal
from g1catch.experiment import RESULTS, Trial, catch_summary, run_trials, write_rows
from g1catch.residual import GpResidual, TreeResidual
from g1catch.runtime import RuntimeConfig
from g1catch.sim import READY_POSTURE, CatchSim, ThrowDistribution

REFERENCE_RATE_HZ = 200.0
REFERENCE_NOISE_M = 0.002
SAVGOL_WINDOW = 41              # 0.2 s: second derivatives of 2 mm noise need a long window
CALIBRATION_THROWS = 150
TEST_THROWS = 200
FORECAST_S = 0.30
ARM_AWAY = READY_POSTURE.copy()
ARM_AWAY[3], ARM_AWAY[4] = 0.5, -1.5      # out of the ball's path
CLOSED_LOOP_THROWS = 120
GP_POINTS = 200                 # inducing set size: the filter evaluates the GP thousands of times per frame


def fly(sim: CatchSim, throw, stop_x: float = 0.45):
    """True ball states at the reference rate until the ball reaches the robot."""
    sim.reset(throw, ARM_AWAY)
    every = int(round(1.0 / (REFERENCE_RATE_HZ * sim.dt)))
    ps, vs = [], []
    tick = 0
    while sim.ball_position[0] > stop_x and sim.ball_position[2] > 0.1 and sim.time < 1.2:
        if tick % every == 0:
            ps.append(sim.ball_position)
            vs.append(sim.ball_velocity)
        sim.step(np.zeros(10))
        tick += 1
    return np.array(ps), np.array(vs)


def calibration_set(seed: int):
    """Velocities and residual accelerations (vs gravity only) from noisy reference positions."""
    rng = np.random.default_rng(seed)
    sim = CatchSim()
    dist = ThrowDistribution()
    dt = 1.0 / REFERENCE_RATE_HZ
    vel, res = [], []
    for _ in range(CALIBRATION_THROWS):
        ps, _ = fly(sim, dist.sample(rng))
        if len(ps) < SAVGOL_WINDOW + 4:
            continue
        noisy = ps + rng.normal(0.0, REFERENCE_NOISE_M, ps.shape)
        v = savgol_filter(noisy, SAVGOL_WINDOW, 3, deriv=1, delta=dt, axis=0)
        a = savgol_filter(noisy, SAVGOL_WINDOW, 3, deriv=2, delta=dt, axis=0)
        trim = SAVGOL_WINDOW // 2
        vel.append(v[trim:-trim])
        res.append(a[trim:-trim] - np.array([0.0, 0.0, -bal.GRAVITY_M_S2]))
    return np.vstack(vel), np.vstack(res)


def open_loop_errors(models: dict, seed: int):
    """Position error after FORECAST_S of forecast from the true state, per model."""
    rng = np.random.default_rng(seed)
    sim = CatchSim()
    dist = ThrowDistribution()
    n = int(FORECAST_S * REFERENCE_RATE_HZ)
    errors = {name: [] for name in models}
    for _ in range(TEST_THROWS):
        ps, vs = fly(sim, dist.sample(rng))
        if len(ps) <= n + 1:
            continue
        start = len(ps) - 1 - n
        for name, (kd, residual) in models.items():
            p, v = ps[start][None, :], vs[start][None, :]
            for _ in range(n):
                p, v = bal.step(p, v, 1.0 / REFERENCE_RATE_HZ, kd, residual)
            errors[name].append(float(np.linalg.norm(p[0] - ps[-1])))
    return {k: np.array(v) for k, v in errors.items()}


def main():
    RESULTS.mkdir(exist_ok=True)
    vel, res = calibration_set(seed=100)
    print(f"calibration: {len(vel)} samples, residual rms {np.sqrt((res**2).sum(1).mean()):.2f} m/s^2")
    t0 = time.perf_counter()
    gp = GpResidual.fit(vel, res, max_points=GP_POINTS)
    t_gp = time.perf_counter() - t0
    t0 = time.perf_counter()
    trees = TreeResidual.fit(vel, res)
    t_tree = time.perf_counter() - t0
    print(f"GP: length scales {np.round(gp.length_scale, 1)} m/s, signal var {gp.signal_variance:.2f}, "
          f"noise var {gp.noise_variance:.2f}, fit {t_gp:.1f} s.  trees fit {t_tree:.1f} s")
    with open(RESULTS / "gp_residual.pkl", "wb") as f:
        pickle.dump(gp, f)

    probe = np.array([[-5.0, 0.0, 0.0], [-5.0, 0.0, -3.0], [-15.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    physics = -bal.drag_per_metre(0.5) * np.linalg.norm(probe, axis=1, keepdims=True) * probe
    print("probe velocity | mean true drag (Cd 0.5) | GP mean | GP std | trees")
    for v, a_true, a_gp, s_gp, a_tr in zip(probe, physics, gp(probe), gp.std(probe), trees(probe)):
        print(f"  {v} | {np.round(a_true, 2)} | {np.round(a_gp, 2)} | {s_gp:.2f} | {np.round(a_tr, 2)}")

    models = {"gravity only": (0.0, None), "nominal drag (physics)": (bal.NOMINAL_KD, None),
              "gravity + GP": (0.0, gp), "gravity + trees": (0.0, trees)}
    errors = open_loop_errors(models, seed=200)
    timing = {}
    batch = np.random.default_rng(0).normal(0, 3, (13, 3))
    for name, fn in (("gravity + GP", gp), ("gravity + trees", trees)):
        t0 = time.perf_counter()
        for _ in range(200):
            fn(batch)
        timing[name] = 1e6 * (time.perf_counter() - t0) / 200
    with open(RESULTS / "residual_open_loop.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "median_cm", "p90_cm", "max_cm", "call_us_13_points"])
        for name, e in errors.items():
            w.writerow([name, len(e), 100 * np.median(e), 100 * np.percentile(e, 90), 100 * e.max(), timing.get(name, "")])
            print(f"{name:26s} {FORECAST_S:.2f} s forecast error: median {100 * np.median(e):.2f} cm, "
                  f"p90 {100 * np.percentile(e, 90):.2f} cm" + (f", {timing[name]:.0f} us per call" if name in timing else ""))

    dist = ThrowDistribution()
    trials = [Trial("gravity only", RuntimeConfig(), dist, CLOSED_LOOP_THROWS, 300, kd=0.0),
              Trial("nominal drag (physics)", RuntimeConfig(), dist, CLOSED_LOOP_THROWS, 300),
              Trial("gravity + GP", RuntimeConfig(), dist, CLOSED_LOOP_THROWS, 300, kd=0.0, residual=gp)]
    out = run_trials(trials)
    rows = []
    for name, r in out.items():
        s = catch_summary(r)
        rows += r
        fc = np.nanmedian([x["forecast_error_cm"] for x in r])
        print(f"closed loop, {name:24s}: caught {s['caught']}/{s['n']} = {100 * s['rate']:.1f} % "
              f"[{100 * s['ci_lo']:.0f}, {100 * s['ci_hi']:.0f}], final forecast error median {fc:.2f} cm")
    write_rows(RESULTS / "residual_closed_loop.csv", rows)


if __name__ == "__main__":
    main()
