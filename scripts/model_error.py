"""What happens when the controller's model of the robot is wrong, and what helps.

The controller's link masses and inertias are scaled by random factors within +/- the stated
error (the simulated robot keeps the true ones). Encoders are 15-bit and velocity comes from an
observer in every condition. Four joint controllers are compared on the same throws:

    computed torque + LQR gains on a Kalman velocity estimate (LQG)
    the same with the extended-state observer's disturbance estimate cancelled
    the same with a sliding-mode term
    both robust terms together

A second sweep puts errors into the camera calibration, where no joint controller can help.

    python scripts/model_error.py
"""
import numpy as np

from g1catch.experiment import RESULTS, Trial, catch_summary, run_trials, write_rows
from g1catch.runtime import RuntimeConfig
from g1catch.sim import ThrowDistribution

THROWS = 80
SEED = 6000
MASS_ERRORS = (0.0, 0.30, 0.50)
VERGENCE_ERRORS_DEG = (0.0, 0.05, 0.1, 0.2, 0.4)
PITCH_ERRORS_DEG = (0.5, 1.0, 2.0)
SLIDING_GAIN = 20.0        # rad/s^2, about the disturbance a 30 % mass error produces in a reach


SOFT_LOOP_WEIGHTS = (4.0e5, 1.0e2)      # LQR weights for a 4 Hz joint loop, a third of the default bandwidth


def controller(mass_error: float, kind: str, soft: bool = False) -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.model_mass_error = mass_error
    cfg.model_seed = 1
    if soft:
        cfg.controller.lqr_position_weight, cfg.controller.lqr_velocity_weight = SOFT_LOOP_WEIGHTS
    if kind in ("disturbance observer", "both"):
        cfg.controller.sensing.observer = "eso"
        cfg.controller.cancel_disturbance = True
    if kind in ("sliding mode", "both"):
        cfg.controller.sliding_gain = SLIDING_GAIN
    return cfg


def camera(vergence_deg: float = 0.0, pitch_deg: float = 0.0) -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.stereo_vergence_error_deg = vergence_deg
    cfg.head_pitch_error_deg = pitch_deg
    return cfg


def report(title: str, trials: list[Trial], csv_name: str) -> None:
    out = run_trials(trials)
    print(f"== {title}", flush=True)
    rows, summary = [], []
    for t in trials:
        r = out[t.name]
        rows += r
        s = catch_summary(r)
        s.update({"trial": t.name,
                  "median_cup_error_cm": float(np.nanmedian([x["cup_error_cm"] for x in r])),
                  "p90_cup_error_cm": float(np.nanpercentile([x["cup_error_cm"] for x in r], 90)),
                  "median_forecast_error_cm": float(np.nanmedian([x["forecast_error_cm"] for x in r])),
                  "median_tracking_mrad": float(np.median([x["reach_tracking_mrad"] for x in r])),
                  "p90_tracking_mrad": float(np.percentile([x["reach_tracking_mrad"] for x in r], 90)),
                  "reach_saturated": float(np.mean([x["saturated_ticks_reach"] > 0 for x in r]))})
        summary.append(s)
        print(f"  {t.name:44s} {s['caught']:3d}/{s['n']} = {100 * s['rate']:5.1f} % [{100 * s['ci_lo']:.0f}, {100 * s['ci_hi']:.0f}]"
              f"  tracking {s['median_tracking_mrad']:.1f} / {s['p90_tracking_mrad']:.1f} mrad"
              f"  cup error {s['median_cup_error_cm']:.2f} / {s['p90_cup_error_cm']:.2f} cm"
              f"  forecast {s['median_forecast_error_cm']:.2f} cm  reach saturated {s['reach_saturated']:.2f}", flush=True)
    write_rows(RESULTS / f"{csv_name}_throws.csv", rows)
    write_rows(RESULTS / f"{csv_name}.csv", summary)


def main():
    dist = ThrowDistribution()
    kinds = ("computed torque (LQG)", "disturbance observer", "sliding mode")
    trials = [Trial(f"{int(100 * e)} % mass error, {k}", controller(e, k), dist, THROWS, SEED) for e in MASS_ERRORS for k in kinds]
    report("link mass error, default 12 Hz joint loop", trials, "model_error_mass")
    trials = [Trial(f"{int(100 * e)} % mass error, {k}", controller(e, k, soft=True), dist, THROWS, SEED)
              for e in MASS_ERRORS for k in kinds]
    report("link mass error, soft 4 Hz joint loop", trials, "model_error_mass_soft")
    trials = [Trial(f"vergence {v} deg", camera(vergence_deg=v), dist, THROWS, SEED) for v in VERGENCE_ERRORS_DEG]
    trials += [Trial(f"head pitch {p} deg", camera(pitch_deg=p), dist, THROWS, SEED) for p in PITCH_ERRORS_DEG]
    report("camera calibration error", trials, "model_error_camera")


if __name__ == "__main__":
    main()
