"""How good are acceleration limits as a stand-in for torque limits?

The reach MPC plans joint accelerations under box limits and leaves torque to the computed-torque
loop. This script takes reach problems from the catch workspace and, for each one,

  1. computes the torque the MPC plan needs by inverse dynamics, and tracks the plan in
     simulation with the real torque limits,
  2. solves the same reach with iLQR on the full arm dynamics with torque limits as hard bounds.

    python scripts/ilqr_check.py
"""
import csv
import time
from pathlib import Path

import mujoco
import numpy as np

from g1catch import ilqr
from g1catch.control import (CONTROL_PERIOD_S, FRICTION_SMOOTHING_RAD_S, JOINT_FRICTION_NM, ArmController,
                             ArmModel, JointTrajectory)
from g1catch.intercept import CatchIk, InterceptConfig
from g1catch.mpc import ReachMpc
from g1catch.sim import ARM_DOF, READY_POSTURE, CatchSim, Throw, ThrowDistribution, load_arm_only

RESULTS = Path(__file__).resolve().parents[1] / "results"
PROBLEMS = 40
ILQR_STEP_S = 0.004
EFFORT_WEIGHT = 1e-3              # on (torque / limit)^2 per step
TERMINAL_POSITION_WEIGHT = 1e4    # per rad^2
TERMINAL_VELOCITY_WEIGHT = 5.0    # per (rad/s)^2


def sample_problem(rng, ik: CatchIk, icfg: InterceptConfig):
    dist = ThrowDistribution()
    point = rng.uniform(dist.aim_lo, dist.aim_hi)
    ball_v = np.array([rng.uniform(-6.0, -4.0), rng.uniform(-1.0, 1.0), rng.uniform(-3.0, 0.5)])
    axis = -ball_v / np.linalg.norm(ball_v)
    q_goal, _, jac = ik.solve(READY_POSTURE, point, axis, iterations=80)
    cup_speed = min(icfg.velocity_match * np.linalg.norm(ball_v), icfg.max_cup_speed_m_s)
    dq_goal = ik.joint_velocity_for(jac, -cup_speed * axis)
    return q_goal, dq_goal, float(rng.uniform(0.22, 0.36)), axis


def cup_lateral_error(arm: ArmModel, q, q_goal, axis) -> float:
    arm.update(q_goal, np.zeros(ARM_DOF))
    target = arm.cup_position
    arm.update(q, np.zeros(ARM_DOF))
    off = arm.cup_position - target
    return float(np.linalg.norm(off - (off @ axis) * axis))


def evaluate_mpc(sim: CatchSim, arm: ArmModel, mpc: ReachMpc, q_goal, dq_goal, horizon, axis):
    sol = mpc.solve(READY_POSTURE, np.zeros(ARM_DOF), q_goal, dq_goal, horizon)
    traj = JointTrajectory(0.0, READY_POSTURE, np.zeros(ARM_DOF), sol.accels, sol.step)
    # Torque the plan asks for, by inverse dynamics along the reference.
    peak = 0.0
    for t in np.arange(0.0, horizon, CONTROL_PERIOD_S):
        q, dq, ddq = traj.sample(t)
        arm.update(q, dq)
        tau = arm.mass_matrix @ ddq + arm.bias + JOINT_FRICTION_NM * np.tanh(dq / FRICTION_SMOOTHING_RAD_S)
        peak = max(peak, float(np.max(np.abs(tau) / sim.torque_limit)))
    # Track it with the real limits.
    sim.reset(Throw(np.array([5.0, 5.0, 0.033]), np.zeros(3)))
    ctl = ArmController(ArmModel(sim.model), sim.torque_limit)
    while sim.time < horizon - 1e-9:
        q_ref, dq_ref, ddq_ref = traj.sample(sim.time)
        tau = ctl.computed_torque(sim.q, sim.dq, q_ref, dq_ref, ddq_ref)
        sim.step(tau)
        sim.step(tau)
    return {"mpc_peak_torque": peak, "mpc_saturated_ticks": ctl.saturated_ticks,
            "mpc_cup_error_cm": 100 * cup_lateral_error(arm, sim.q, q_goal, axis),
            "mpc_velocity_error": float(np.linalg.norm(sim.dq - dq_goal))}


def evaluate_ilqr(model, data, arm: ArmModel, limit, q_goal, dq_goal, horizon, axis):
    nx = 2 * ARM_DOF
    a = np.zeros((nx, nx))
    b = np.zeros((nx, ARM_DOF))

    def step(x, u):
        data.qpos[:], data.qvel[:], data.ctrl[:] = x[:ARM_DOF], x[ARM_DOF:], u
        data.time = 0.0
        mujoco.mj_step(model, data)
        return np.concatenate([data.qpos, data.qvel])

    def linearise(x, u):
        data.qpos[:], data.qvel[:], data.ctrl[:] = x[:ARM_DOF], x[ARM_DOF:], u
        mujoco.mjd_transitionFD(model, data, 1e-6, True, a, b, None, None)
        return a.copy(), b.copy()

    n = int(round(horizon / ILQR_STEP_S))
    arm.update(READY_POSTURE, np.zeros(ARM_DOF))
    hold = np.tile(arm.bias, (n, 1))                       # start from gravity compensation
    x0 = np.concatenate([READY_POSTURE, np.zeros(ARM_DOF)])
    x_goal = np.concatenate([q_goal, dq_goal])
    qf = np.diag([TERMINAL_POSITION_WEIGHT] * ARM_DOF + [TERMINAL_VELOCITY_WEIGHT] * ARM_DOF)
    t0 = time.perf_counter()
    res = ilqr.solve(step, linearise, x0, hold, x_goal, EFFORT_WEIGHT / limit**2, qf, -limit, limit)
    return {"ilqr_iterations": res.iterations, "ilqr_converged": int(res.converged),
            "ilqr_solve_s": time.perf_counter() - t0,
            "ilqr_peak_torque": float(np.max(np.abs(res.controls) / limit)),
            "ilqr_cup_error_cm": 100 * cup_lateral_error(arm, res.states[-1, :ARM_DOF], q_goal, axis),
            "ilqr_velocity_error": float(np.linalg.norm(res.states[-1, ARM_DOF:] - dq_goal))}


def main():
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    sim = CatchSim()
    arm = ArmModel(sim.model)
    ik = CatchIk(sim.model, sim.joint_lo, sim.joint_hi)
    mpc = ReachMpc(sim.joint_lo, sim.joint_hi)
    arm_model = load_arm_only(ILQR_STEP_S)
    arm_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT   # self-collision is not optimised over
    arm_data = mujoco.MjData(arm_model)
    rows = []
    for i in range(PROBLEMS):
        q_goal, dq_goal, horizon, axis = sample_problem(rng, ik, InterceptConfig())
        row = {"problem": i, "horizon_s": horizon}
        row.update(evaluate_mpc(sim, arm, mpc, q_goal, dq_goal, horizon, axis))
        row.update(evaluate_ilqr(arm_model, arm_data, arm, sim.torque_limit, q_goal, dq_goal, horizon, axis))
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(RESULTS / "ilqr_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    col = lambda k: np.array([r[k] for r in rows])
    print(f"\nMPC plans whose inverse-dynamics torque exceeds a limit: {np.mean(col('mpc_peak_torque') > 1.0):.2f}")
    print(f"MPC plans that saturated while tracking:                {np.mean(col('mpc_saturated_ticks') > 0):.2f}")
    print(f"cup error at the catch time, median / p90: MPC + computed torque {np.median(col('mpc_cup_error_cm')):.2f} / "
          f"{np.percentile(col('mpc_cup_error_cm'), 90):.2f} cm, iLQR {np.median(col('ilqr_cup_error_cm')):.2f} / "
          f"{np.percentile(col('ilqr_cup_error_cm'), 90):.2f} cm")
    print(f"terminal joint-velocity error, median: MPC {np.median(col('mpc_velocity_error')):.2f} rad/s, "
          f"iLQR {np.median(col('ilqr_velocity_error')):.2f} rad/s")
    print(f"iLQR: {np.median(col('ilqr_iterations')):.0f} iterations and {np.median(col('ilqr_solve_s')):.1f} s median, "
          f"converged {np.mean(col('ilqr_converged')):.2f}")


if __name__ == "__main__":
    main()
