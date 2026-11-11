"""Catching from rendered stereo images: the CNN detector against the colour threshold.

Every frame is rendered, blurred along the ball's image motion, and passed to the detector. The
same seeded throws are also run with the pixel-noise sensor that the other experiments use, to
show how well that stand-in represents the real pipeline.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python scripts/vision_in_loop.py
"""
import numpy as np

from g1catch.experiment import RESULTS, Trial, catch_summary, run_trials, write_rows
from g1catch.runtime import RuntimeConfig
from g1catch.sim import ThrowDistribution

THROWS = 60
SEED = 500


def main():
    dist = ThrowDistribution()
    trials = [Trial("pixel-noise model", RuntimeConfig(), dist, THROWS, SEED)]
    for sensor in ("cnn", "blob"):
        for look in ("nominal", "randomized", "hard"):
            trials.append(Trial(f"{sensor}, {look}", RuntimeConfig(), dist, THROWS, SEED, sensor=sensor, appearance=look))
    out = run_trials(trials)
    rows = []
    for name, r in out.items():
        s = catch_summary(r)
        rows += r
        committed = np.mean([x["committed"] for x in r])
        fc = np.nanmedian([x["forecast_error_cm"] for x in r])
        print(f"{name:20s} caught {s['caught']:3d}/{s['n']} = {100 * s['rate']:5.1f} % [{100 * s['ci_lo']:.0f}, {100 * s['ci_hi']:.0f}]"
              f"  committed {100 * committed:5.1f} %  forecast error at catch {fc:.2f} cm", flush=True)
    write_rows(RESULTS / "vision_in_loop.csv", rows)


if __name__ == "__main__":
    main()
