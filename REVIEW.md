# Code Review: g1-catch

Reviewed as a staff engineer on a humanoid controls and perception team would review a pull request. This is the second review of the repository. The first found seven behavioural defects in the original pipeline (all fixed, listed in section 3). Its open items asked for an imperfect robot model, real-time planning, a tested bounce strategy, collision checking, constraints on the learned policy, saturation logging and a tuning discipline. This pass reviews the changes that answer those: a C++ flight core, joint observers and robust terms, model and calibration error, a planning delay, self-collision checks, a barrier-function safety filter, a multi-candidate track start and a tuning script. Scope: `g1catch/`, `cpp/`, `scripts/`, `tests/` and the build files.

## 1. Overall assessment

The second pass did what a good second pass should. It put the idealisations under test, and most of them turned out not to matter. Joint sensing, link-mass error up to 50 % and stereo vergence error up to 0.4° leave the catch rate unchanged. The loop analysis explains why before any simulation is run: the LQG joint loop has a gain margin of 9, and a mass error is a gain error. The robust controllers added for model error work as designed (they halve tracking error at a soft loop) and change nothing downstream, because joint tracking contributes about half a millimetre to a 14 mm cup error. The README says so plainly. That is the right outcome, and the experiments that show it are the most useful thing in this pass.

Three changes did move the task metric. The upward aim floor took bounce catches from 5 to 21 of 120. The multi-candidate track start took rendered catching under randomized appearance from 73 % to 83 % and under the hardest condition from 38 % to 48 %. The C++ core and solver reuse took re-planning from about 23 ms to 2.1 ms per frame. Three defects were found by the new experiments themselves, the worst of which made any planning delay longer than a frame period freeze the arm.

Two things weaken the pass. The baseline fell from 89 % to 83 % between versions and none of the realism ablations accounts for it. Until that is bisected, the README's headline number has an unexplained 6-point history. And the planner's compute time is still assumed rather than enforced: the commit call takes 15 ms against a 10 ms assumption, and the simulator's clock waits for it.

## 2. Scorecard

| Area | Score | Basis |
|---|---|---|
| Correctness | 8 | First-pass checks plus a compiled core held to its reference at 1e-12, observer error dynamics and gains checked against the Riccati solution, a barrier filter tested against adversarial commands, loop margins predicted and confirmed in simulation for three designs. Three new defects found and fixed in this pass. One unexplained regression (section 1). |
| Architecture | 8 | Observers, robust terms, the safety filter and the error injection each enter through one config field and are off or neutral by default, so every ablation is a one-line change. `JointObserver` is shared by both control laws and fed the saturated acceleration. |
| Readability | 7 | `_ThrowState` keeps the runtime readable after the additions. The pending-plan queue is a tuple deque where a small dataclass would read better. `RuntimeConfig` now carries robot-error fields next to planner fields. |
| Self-documentation | 8 | New modules open with what they model and why. Constants carry units and the measurement behind them (the observer's 28 ms rise time, the 20° aim floor). |
| Maintainability | 7 | Per-group CSV files now rebuild the summary, so groups can be rerun alone. The compiled core has a pure-Python fallback, so a missing compiler degrades speed and not behaviour. `Trial` still has 18 fields. |
| Testing | 8 | 71 tests, including a rendered end-to-end test that pins the detector weights (8 of 10 nominal throws), and tests that fail on each defect fixed in this pass. Learned components are still tested only by smoke runs. |
| Performance | 6 | Re-planning 2.1 ms median, 12.7 ms at the 99th percentile. The commit call is 15 ms median against an assumed 10 ms. Only the flight model is compiled. The simulator does not charge overruns as latency. |
| Reliability | 7 | Lost tracks are restarted from any numerical failure in perception or planning, plans are queued, the safety filter has a closed-form bound with no iteration count. Robot bodies still touch in 4.3 % of throws. |
| API design | 8 | `PolicyPlanner` keeps the `ReachMpc.solve` interface with the filter inside it. `filter_acceleration` is a pure function. `model_with_mass_error` returns a copy and leaves the simulator's model alone, which a test checks. |
| Domain-specific correctness | 8 | Quantization modelled as noise of variance $\Delta^2/12$, observer fed the post-saturation acceleration so a torque limit is not mistaken for a disturbance, braking-distance barrier rather than a position barrier, calibration error applied to the tracker's belief and not to the images. |
| Safety and robustness | 6 | Barrier filter behind the learned planner, collision refusal at the catch configuration, measured margins for every loop design. No path or post-catch collision checking, no supervisor, kinematic errors untested. |

## 3. Problems

### Fixed in this pass

| Severity | Location | What | Why | Fix and evidence |
|---|---|---|---|---|
| High | `runtime.py`, planning delay | One pending-plan slot. With a delay longer than the 16.5 ms frame period, each new plan replaced the previous one before it took effect. | No plan was ever applied and the arm stayed at rest. A 20 ms delay caught 5 of 120. | Plans wait in a queue ordered by start time, and the planner starts from the newest queued plan. Same throws: 94 of 120. The realism group now runs 10, 20 and 35 ms. |
| Medium | `runtime.py`, `imm.py` | With the outlier gate off, a false detection could make the covariance too large for an eigendecomposition, and the run raised. | The gate-off ablation could not be run, so the gate's value was unmeasured. | A numerical failure anywhere in perception or planning drops the track and restarts it. The ablation now runs: 97 against 14 of 120. |
| Medium | `joints.py` | An observer used before `reset` started from zero joint angles. | The first control tick saw an error equal to the whole posture and commanded torque accordingly. Found when the full test suite first ran on the new controller. | The observer starts from its first reading if it was not reset. |
| Medium | `intercept.py`, bounce catches | The cup was aimed straight against the ball's velocity. The first review inferred, without testing, that this fails for a rising ball. | A rising ball asks the cup to open downward. 3 of 120 bounce passes caught. | Aim never below 20° above horizontal, rising catch points penalised. 21 of 120 against 5 of 120 for the straight aim on the same throws (p = 1e-4). The ball now passes within 3 cm of the cup centre less often (29 % against 53 %), so this traded reach for retention. |
| Medium | `detector.py`, `imm.py`, track start | One detection per image, the strongest. | A brighter ball-coloured object in the background hid the ball and the track never started. | Up to three peaks per image with non-maximum suppression, and the tracker starts from the left-right pair that triangulates inside the start volume with the smallest reprojection error. CNN on 60 rendered throws: 73 % to 83 % under randomized appearance, 38 % to 48 % in the hardest condition. `test_track_starts_on_the_ball_when_a_brighter_decoy_is_listed_first`. |
| Medium | `ukf.py`, `imm.py`, `mpc.py`, `intercept.py`: planner time | The planner took about 23 ms per frame against a 16.5 ms period. | The results assumed a planner that does not exist. | Flight model in C++ (45 times faster on a one-second forecast, `tests/test_flight_core.py` holds it to the NumPy reference at 1e-12), one OSQP solver per horizon length updated in place, IK stops when the step is below 1e-5 rad. Re-planning median 2.1 ms. The commit call is still over budget (open items). |
| Medium | `safety.py` | The first version of the barrier filter found its bound by 24 steps of bisection. | 12 ms per plan behind a policy whose point is to be fast, nine times its own planning time. | Closed-form largest root of the stopping-distance quadratic. 1.6 ms per plan, same violation rate. `test_filtered_joint_never_leaves_its_range_under_an_adversarial_command`. |
| Low | `scripts/margins.py` | The phase margin was read at the last unit-gain crossing after unwrapping. | The disturbance-observer loop crosses unit gain more than once, and the result depended on where unwrapping started. | Phase lag available before −180° taken modulo 360 at every crossing, and the crossing with the least delay tolerance reported. Predicted delay margins now match the simulation for all three designs. |
| Low | `scripts/run_experiments.py` | Running one group rewrote `exp_summary.csv` with that group only. | Figures built from the summary lost the other groups. | The summary is rebuilt from every per-group CSV on disk after each run. |
| Low | build | Compiled `.so` and `build/` files were committed. | A binary built for one Python ABI in the repository, and a stale one would shadow a rebuild. | Untracked, ignored and excluded from the Docker context. |

### Fixed in the first review

The first review fixed seven behavioural defects. They are still covered by the tests named in each row.

| Severity | What | Test |
|---|---|---|
| High | The catch time was re-optimised every frame and crept along the ball's path. | `test_catch_time_is_not_reoptimised_every_frame` |
| High | A posture term leaked through damped null-space projectors and held the cup 1.5 cm off target. | `test_ik_puts_the_cup_on_target_and_aims_it` |
| High | Impedance gains were multiplied by task inertia, so the physical stiffness was not what the parameters said. | `test_impedance_is_soft_along_the_axis_and_stiff_across_it` |
| High | A stale plan was kept when the ball left the reachable set late in the throw. | End-to-end tests and the baseline's 2.4 cm 90th-percentile forecast error |
| Medium | A single IMM transition matrix inflated uncertainty for an impact in mid-air. | `test_coasting_in_open_air_does_not_assume_an_impact` |
| Medium | The covariance could lose positive definiteness. | `test_symmetric_positive_repairs_an_indefinite_covariance` |
| Medium | MuJoCo's derived constants were stale after a mass change. | `test_heavier_ball_rests_in_the_cup_without_exciting_the_contact` |

### Open

| Severity | Location | What | Why | Recommended fix |
|---|---|---|---|---|
| High | baseline | 267 of 300 caught in the first version, 249 of the same 300 now. The realism ablations each move the rate by at most 3 of 120. | The headline number changed by 6 points for a reason nobody has identified. It could be the aim floor, the rising-point penalty, the collision refusal, or an interaction. | Bisect: run the 300 baseline throws with each change of this pass switched off in turn, and report the result whichever way it falls. |
| High (hardware), Medium (simulation) | `runtime.py` | The commit call takes 15 ms median, 38 ms worst, against the assumed 10 ms. The simulator's clock waits for the planner. | Overruns are free in simulation. The measured 20 ms delay costs nothing, so the fix is cheap, but the default should match the measurement. | Set the default planning delay to 20 ms, and run the planner on its own clock so an overrun appears as latency. Then budget the commit-time IK. |
| High (hardware) | planner | Collision checking covers the catch configuration only. Robot bodies touched in 13 of 300 baseline throws, against 14 before the catch-configuration check. | The paths to and from the catch are where the contacts happen. | Capsule distances as constraints along the MPC trajectory and in the secure move. |
| Medium | `runtime.py`, secure move | A torque limit was active after contact in 83 % of throws, at the shoulder pitch joint in 199 of 248. None during the reach. | The post-catch brake and upright move are more aggressive than they need to be, and saturating there after contact risks throwing the ball out. | Lower acceleration limits on the secure move and check whether catches change. |
| Medium | bounce throws | 21 of 120. The cup now misses by 4 cm in the median. | The aim floor fixed retention and exposed the reach. | Prefer catch points after the apex, or allow a later commit for bounce throws when the impact mode is active. |
| Medium | model error | Link lengths, joint zero offsets, camera intrinsics and stereo baseline are exact. | These move the cup and the depth estimate directly. Mass error, which was tested, turned out not to matter. | Perturb them the same way and report catch rate against each. |
| Medium | perception | 48 % in the hardest appearance condition. | Background clutter still starts or holds wrong tracks. | Two-frame input to the CNN, or a size-against-depth check after triangulation. |
| Low | `residual.py` | A learned residual forces the tracker off its compiled path. | The GP that caught more on one batch cannot be used in real time as written. | Evaluate the GP's kernel sum in the C++ core, or fit a parametric drag term from the same data. |
| Low | `ppo.py` | Exploration noise stayed at 0.30 for both runs. | The policy never converged, so its null result says little. | A separate, faster learning rate for the log standard deviation, a lower entropy weight, a larger throw budget. |
| Low | `ilqr.py` | Hit the iteration cap on all 40 problems. | Unchanged from the first review. | Projected-Newton box QP in the backward pass. |
| Low | `experiment.py` | `Trial` has 18 fields. | Unchanged. | Split into sensor, model and planner options. |
| Low | learned components | One seed each. | Unchanged. | Three seeds each. |
| Low | `sensing.py`, `SyntheticSensor` | 85 % against 97 % for the rendered CNN pipeline on the same 60 throws. | The stand-in is pessimistic, which is the safe direction, but it is not calibrated. | Fit its noise and miss rate by range to logged detector errors. |

## 4. Readability and self-documentation

The planning-delay bug, before and after. The first version is the natural thing to write and is wrong as soon as the delay exceeds a frame.

```python
# before: one slot
st.pending_traj, st.pending_q_final = self._with_follow_through(start, q_ref, dq_ref, sol)
st.pending_from = start
...
if st.pending_traj is not None and now >= st.pending_from:
    st.traj, st.q_final, st.pending_traj = st.pending_traj, st.pending_q_final, None

# after: a queue in start-time order
# Plans that have been computed and not yet taken effect: (start time, trajectory, final posture).
# More than one can be waiting when the planning delay is longer than a camera frame.
pending: deque = field(default_factory=deque)
...
st.pending.append((start, *self._with_follow_through(start, q_ref, dq_ref, sol)))
...
while st.pending and now >= st.pending[0][0]:
    _, st.traj, st.q_final = st.pending.popleft()
```

The safety filter, before and after. The closed form is shorter and its docstring carries the derivation.

```python
# before: 24 bisection steps per limit per plan step
for _ in range(BISECTION_STEPS):
    mid = 0.5 * (upper_ok_lo + upper_hi)
    safe = _stopping_position(q, dq, mid, step, a_brake, +1.0) <= q_hi
    ...

# after
def _fastest_safe_velocity(gap, dq, step, a_brake):
    """Largest velocity toward a limit, one step from now, from which the joint can still stop.
        step (dq + w) / 2 + w^2 / (2 a_brake) <= gap     for w >= 0.
    """
u_max = np.clip((_fastest_safe_velocity(q_hi - q, dq, step, a_brake) - dq) / step, lo, hi)
```

What still reads poorly: the pending-plan tuple (a three-field dataclass would name its parts), and `RuntimeConfig`, which now mixes what the robot is (mass error, calibration error) with what the controller does.

## 5. Architecture and design

The error-injection design is right. Mass error lives in the controller's copy of the model and calibration error in the poses the tracker believes. The simulated robot and the rendered images stay true, so every experiment is "a correct world seen through a wrong model", which is the situation on hardware.

The observer is fed the acceleration that the saturated torque will produce, not the one that was asked for. Without that, every torque limit would look like a disturbance and the extended-state observer would fight the limit. This is easy to get wrong and is done correctly.

The safety filter is a pure function between the learned planner and the trajectory. It never changes a safe command, so behind the MPC it would be a no-op, and behind the network it is the only source of constraint satisfaction. That is the correct placement for a learned component.

The C++ core is the right first piece to move. It was most of the planner's time, it has no dependencies, and the NumPy version remains as a reference that the tests hold it to.

## 6. Testing and verification

New claims in this pass:

| Claim | Verified by |
|---|---|
| The compiled flight model equals the NumPy reference, including through bounces | Tests, to 1e-12 and 1e-10 |
| Observer error dynamics are stable and the gains solve the filter Riccati equation | Test (math) |
| The observers recover velocity from a 15-bit encoder, and the ESO reaches 60 % of a constant disturbance within 60 ms and its mean within 3 % | Tests in simulation |
| A filtered joint never leaves its range under a command pushing it at full acceleration | Test, 200 random start states |
| The wrong-mass model changes dynamics, keeps kinematics, and leaves the simulator's model alone | Test |
| A folded arm is refused as a catch configuration | Test |
| A brighter decoy listed first does not start the track | Test |
| A vergence error biases the forecast | End-to-end test |
| The rendered pipeline with the trained CNN catches at least 8 of 10 nominal throws | End-to-end test, `render` marker |
| Delay margins of the three loop designs match the simulated arm | `scripts/margins.py` (8.9 / 8.5 / 6.5 ms predicted, stable through 8 / 8 / 6 ms) |
| Mass error up to 50 % and vergence error up to 0.4° leave the catch rate unchanged | Seeded experiments, 80 throws per cell |
| The barrier filter brings joint-range violations of the learned planner to the MPC's level | 300 random reach problems |
| Re-planning fits a 10 ms budget and the commit call does not | `scripts/planner_timing.py`, 80 throws on an idle machine |
| The Docker image builds the C++ core and passes the tests | Built and run here: 71 passed in 121 s inside the container, including the C++ build and the rendered end-to-end test. The build context was a copy with this sandbox's proxy certificate added, so the Dockerfile as committed has not been built on an open network. |
| The GitHub Actions workflow | Not run. It has not been pushed. |
| Anything on a robot | Not verified |

The first review's verification table still holds. Its claims are covered by the same tests.

## 7. Prioritized action plan

Must fix before commit: none outstanding.

Before sharing as a portfolio piece:
1. Bisect the 89 % to 83 % baseline change and state the cause in the README.
2. Make 20 ms the default planning delay, since it is what the commit call needs and it measurably costs nothing.

Recommended:
3. Run the planner on its own clock so overruns become latency.
4. Path and post-catch collision constraints.
5. Gentler secure move, then check catches.
6. Kinematic and intrinsic calibration error.
7. The GP residual in the compiled core.

Optional:
8. A second bounce strategy (catch after the apex).
9. Two-frame detector input.
10. PPO with a learned exploration schedule and more throws, three seeds for every learned component, a smaller `Trial`.

## 8. Final verdict

| Context | Verdict |
|---|---|
| Personal or portfolio project | STRONG. The second pass tested the assumptions the first left open, and reports that most of them did not matter, which is a result. Item 1 should be done before showing it. |
| Shared codebase | ACCEPTABLE, after items 1 and 2 and the `Trial` split. |
| Simulation study | STRONG for direct throws, including model and calibration error. Weak for bounce passes. |
| Hardware | REQUEST CHANGES. Blocking: a planner on its own clock with a measured budget, path collision checking, kinematic calibration error, detector results on real images, and a catcher that does not depend on a soft liner. |
