"""Stability margins of the 1 kHz joint loop for three designs, checked against simulation.

Computed torque leaves each joint as q'' = u. The three designs are

    state feedback   u = -kp q - kd q' on the true state (not available on a real joint)
    LQG              the same gains on a Kalman estimate from the quantized angle
    ESO              LQG plus an estimated disturbance that is cancelled

For each one the script computes the loop transfer broken at the plant input and reports
crossover, phase margin, gain margins in both directions, delay margin and the sensitivity peak.
The gain margins are also tolerances on inertia error: if the true inertia is the model's times
1/g, the loop gain is multiplied by g. The delay margin is then checked on the simulated arm by
adding whole milliseconds of actuation delay until the loop goes unstable.

    python scripts/margins.py
"""
import csv
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from g1catch.control import CONTROL_PERIOD_S, ArmController, ArmModel, ControllerConfig, lqr_joint_gains
from g1catch.joints import JointSensingConfig, kalman_gain, observer_model
from g1catch.sim import ARM_DOF, READY_POSTURE, CatchSim, Throw
from plotstyle import MUTED, SERIES, apply_style

RESULTS = Path(__file__).resolve().parents[1] / "results"
STEP_RAD = 0.05
UNSTABLE_ERROR_RAD = 0.5
DESIGNS = ("state feedback", "LQG", "ESO")


def plant(z: np.ndarray, period: float):
    """Double integrator under zero-order hold: transfer functions from u to q and to q'."""
    return period**2 * (z + 1) / (2 * (z - 1) ** 2), period / (z - 1)


def loop_transfer(design: str, kp: float, kd: float, omega: np.ndarray, period: float) -> np.ndarray:
    z = np.exp(1j * omega * period)
    to_q, to_dq = plant(z, period)
    if design == "state feedback":
        return kp * to_q + kd * to_dq
    with_d = design == "ESO"
    a, b, c = observer_model(period, with_d)
    gain = kalman_gain(period, JointSensingConfig(), with_d)
    k = np.array([kp, kd, 1.0]) if with_d else np.array([kp, kd])
    n = len(k)
    # Predictor-corrector observer with u = -k x_corrected:
    #   x_c = (I - L C) x_p + L y,   x_p+ = (A - B k)(x_c)
    correct = np.eye(n) - np.outer(gain, c[0])
    closed = a - np.outer(b, k)
    controller = np.empty(len(z), dtype=complex)
    for i, zi in enumerate(z):
        x_p = np.linalg.solve(zi * np.eye(n) - closed @ correct, closed @ gain)
        controller[i] = k @ (correct @ x_p + gain)
    return to_q * controller


def margins(loop: np.ndarray, omega: np.ndarray) -> dict:
    mag, phase = np.abs(loop), np.unwrap(np.angle(loop))
    # A loop with integral action can cross unit gain more than once. The margin that matters is
    # at the crossover that tolerates the least delay. Phase lag available before -180 deg is
    # taken modulo 360 so the result does not depend on where the unwrapping started.
    crossings = np.nonzero(np.diff(np.sign(mag - 1.0)))[0]
    lag_room = np.mod(np.angle(loop[crossings]) + np.pi, 2 * np.pi)
    worst = int(np.argmin(lag_room / omega[crossings]))
    i_c = crossings[worst]
    phase_margin = lag_room[worst]
    # Phase crossings of -180 deg (and its multiples): gain there bounds how far the gain may move.
    wrapped = np.angle(-loop)                               # zero when the loop phase is -180 deg
    idx = np.nonzero((np.diff(np.sign(wrapped)) != 0) & (np.abs(wrapped[:-1]) < 1.0))[0]
    gains = mag[idx]
    up = 1.0 / gains[gains < 1.0].max() if np.any(gains < 1.0) else np.inf
    down = 1.0 / gains[gains > 1.0].min() if np.any(gains > 1.0) else 0.0
    return {"crossover_hz": omega[i_c] / (2 * np.pi), "phase_margin_deg": float(np.degrees(phase_margin)),
            "gain_margin_up": float(up), "gain_margin_down": float(down),
            "delay_margin_ms": float(1e3 * phase_margin / omega[i_c]),
            "sensitivity_peak": float(np.max(np.abs(1.0 / (1.0 + loop)))), "mag": mag, "phase": phase}


def controller_config(design: str) -> ControllerConfig:
    cfg = ControllerConfig()
    cfg.sensing.observer = {"state feedback": "exact", "LQG": "kalman", "ESO": "eso"}[design]
    cfg.cancel_disturbance = design == "ESO"
    return cfg


def simulate_with_delay(design: str, delay_ms: int, seconds: float = 1.5) -> float:
    """Peak joint error late in a small step response with the torque delayed. inf if unstable."""
    sim = CatchSim()
    sim.reset(Throw(np.array([5.0, 5.0, 0.033]), np.zeros(3)))
    ctl = ArmController(ArmModel(sim.model), sim.torque_limit, controller_config(design))
    ctl.reset(READY_POSTURE)
    q_ref = READY_POSTURE.copy()
    q_ref[3] += STEP_RAD
    q_ref[6] += STEP_RAD
    zero = np.zeros(ARM_DOF)
    queue = deque([ctl.computed_torque(sim.q, sim.dq, READY_POSTURE, zero, zero)] * delay_ms)
    worst_late = 0.0
    for k in range(int(seconds / CONTROL_PERIOD_S)):
        queue.append(ctl.computed_torque(sim.q, sim.dq, q_ref, zero, zero))
        tau = queue.popleft()
        sim.step(tau)
        sim.step(tau)
        err = float(np.abs(sim.q - q_ref).max())
        if err > UNSTABLE_ERROR_RAD or not np.isfinite(err):
            return float("inf")
        if k * CONTROL_PERIOD_S > seconds - 0.3:
            worst_late = max(worst_late, err)
    return worst_late


def main():
    RESULTS.mkdir(exist_ok=True)
    cfg = ControllerConfig()
    kp, kd = lqr_joint_gains(cfg.lqr_position_weight, cfg.lqr_velocity_weight)
    omega = np.logspace(0, np.log10(np.pi / CONTROL_PERIOD_S * 0.9999), 60_000)
    print(f"kp {kp:.0f} 1/s^2, kd {kd:.1f} 1/s")
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
    rows, sweep = [], []
    for i, design in enumerate(DESIGNS):
        m = margins(loop_transfer(design, kp, kd, omega, CONTROL_PERIOD_S), omega)
        if design == "state feedback":
            # This loop reaches -180 deg only at the Nyquist frequency, where z = -1 and L = -kd T / 2.
            m["gain_margin_up"] = 2.0 / (kd * CONTROL_PERIOD_S)
        last_stable = -1
        for delay in range(0, 15):
            err = simulate_with_delay(design, delay)
            sweep.append((design, delay, err))
            if np.isfinite(err) and err < 0.005 and last_stable == delay - 1:
                last_stable = delay
        row = {"design": design, **{k: v for k, v in m.items() if k not in ("mag", "phase")},
               "simulated_stable_through_ms": last_stable}
        rows.append(row)
        down = f"x{m['gain_margin_down']:.2f}" if m["gain_margin_down"] > 0 else "any"
        print(f"{design:15s} crossover {m['crossover_hz']:5.1f} Hz  phase margin {m['phase_margin_deg']:5.1f} deg  "
              f"gain x{m['gain_margin_up']:.1f} up, {down} down  delay margin {m['delay_margin_ms']:.2f} ms  "
              f"Ms {m['sensitivity_peak']:.2f}  simulation settles through {last_stable} ms of added delay", flush=True)
        hz = omega / (2 * np.pi)
        axes[0].semilogx(hz, 20 * np.log10(m["mag"]), color=SERIES[i], lw=2, label=design)
        axes[1].semilogx(hz, np.degrees(m["phase"]), color=SERIES[i], lw=2)
    axes[0].axhline(0, color=MUTED, lw=1, ls=":")
    axes[0].set_ylabel("magnitude (dB)")
    axes[0].set_ylim(-40, 60)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Joint loop transfer for three designs", fontsize=9)
    axes[1].axhline(-180, color=MUTED, lw=1, ls=":")
    axes[1].set_ylabel("phase (deg)")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_xlim(1, 500)
    axes[1].set_ylim(-360, 0)
    fig.tight_layout()
    fig.savefig(RESULTS / "margins.png", dpi=160)
    with open(RESULTS / "margins.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(RESULTS / "delay_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "delay_ms", "residual_error_rad"])
        w.writerows(sweep)


if __name__ == "__main__":
    main()
