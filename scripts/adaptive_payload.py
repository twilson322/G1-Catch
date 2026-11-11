"""Payload adaptation after a catch: compliant hold with and without the mass estimate.

A ball of unknown mass rests in the upright cup while the arm holds position with a soft
task-space impedance. Without the estimate the cup sags by m g / (task stiffness). Recursive
least squares on the wrist force sensor identifies the mass and the arm model takes it over.

    python scripts/adaptive_payload.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from g1catch.adaptive import PayloadEstimator
from g1catch.control import CONTROL_PERIOD_S, ArmController, ArmModel, ControllerConfig, ImpedanceGains
from g1catch.intercept import CatchIk
from g1catch.sim import ARM_DOF, READY_POSTURE, CatchSim, Throw
from plotstyle import MUTED, SERIES, apply_style

RESULTS = Path(__file__).resolve().parents[1] / "results"
MASSES_KG = (0.057, 0.15, 0.30, 0.50)
HOLD_STIFFNESS = 400.0            # N/m, isotropic: a compliant carry
ADAPT_FROM_S = 1.5
DURATION_S = 4.0
FORCE_NOISE_N = 0.05
ACCEL_FILTER_HZ = 20.0
HOLD_POINT = np.array([0.35, -0.20, 1.10])


def run(mass: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    sim = CatchSim()
    ik = CatchIk(sim.model, sim.joint_lo, sim.joint_hi)
    q_hold, _, _ = ik.solve(READY_POSTURE, HOLD_POINT, np.array([0.0, 0.0, 1.0]), iterations=100)
    sim.reset(Throw(np.array([5.0, 5.0, 0.033]), np.zeros(3), mass=mass), q_hold)
    # Rest the ball on the cup's base.
    sim.data.qpos[ARM_DOF:ARM_DOF + 3] = sim.cup_position - 0.001 * sim.cup_axis
    mujoco.mj_forward(sim.model, sim.data)

    gains = ImpedanceGains(along_axis_n_per_m=HOLD_STIFFNESS, lateral_n_per_m=HOLD_STIFFNESS, along_axis_damping_ratio=1.0)
    arm = ArmModel(sim.model)
    ctl = ArmController(arm, sim.torque_limit, ControllerConfig(impedance=gains))
    est = PayloadEstimator()
    x_ref, axis_ref = sim.cup_position, sim.cup_axis
    site = sim.model.site("ft_site").id
    alpha = 1.0 - np.exp(-2 * np.pi * ACCEL_FILTER_HZ * CONTROL_PERIOD_S)
    prev_vel = np.zeros(3)
    accel = np.zeros(3)
    rows = []
    for k in range(int(DURATION_S / CONTROL_PERIOD_S)):
        t = k * CONTROL_PERIOD_S
        tau = ctl.impedance(sim.q, sim.dq, x_ref, axis_ref, np.zeros(3), np.zeros(3), q_hold, np.zeros(ARM_DOF))
        sim.step(tau)
        sim.step(tau)
        vel = sim.cup_velocity()
        accel += alpha * ((vel - prev_vel) / CONTROL_PERIOD_S - accel)
        prev_vel = vel
        force_site = sim.data.sensor("ft_force").data + rng.normal(0.0, FORCE_NOISE_N, 3)
        force_world = sim.data.site_xmat[site].reshape(3, 3) @ force_site
        mass_hat = est.update(force_world, accel)
        if t >= ADAPT_FROM_S:
            arm.payload_mass = mass_hat
        rows.append((t, mass, mass_hat, 1e3 * (x_ref[2] - sim.cup_position[2]), int(sim.ball_in_cup())))
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    all_rows = []
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for i, mass in enumerate(MASSES_KG):
        rows = np.array(run(mass))
        all_rows += rows.tolist()
        before = rows[(rows[:, 0] > ADAPT_FROM_S - 0.3) & (rows[:, 0] < ADAPT_FROM_S), 3].mean()
        after = rows[rows[:, 0] > DURATION_S - 0.3, 3].mean()
        final = rows[rows[:, 0] > DURATION_S - 0.3, 2].mean()
        print(f"mass {mass:.3f} kg: estimate {final:.3f} kg ({100 * (final - mass) / mass:+.1f} %), "
              f"sag {before:.2f} mm before adaptation, {after:.2f} mm after, ball kept {bool(rows[-1, 4])}")
        axes[0].plot(rows[::10, 0], rows[::10, 2], color=SERIES[i], lw=1.6, label=f"{mass:g} kg")
        axes[0].axhline(mass, color=SERIES[i], lw=0.8, ls=":")
        axes[1].plot(rows[::10, 0], rows[::10, 3], color=SERIES[i], lw=1.6, label=f"{mass:g} kg")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("estimated ball mass (kg)")
    axes[0].set_title("RLS estimate from the wrist force sensor (dotted: true)", fontsize=9)
    axes[1].axvline(ADAPT_FROM_S, color=MUTED, lw=1, ls=":")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("cup sag (mm)")
    axes[1].set_title("Compliant hold: the model takes the estimate at 1.5 s", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, title="ball mass", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "adaptive_payload.png", dpi=160)
    with open(RESULTS / "adaptive_payload.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "mass_kg", "mass_estimate_kg", "sag_mm", "ball_in_cup"])
        w.writerows(all_rows[::10])


if __name__ == "__main__":
    main()
