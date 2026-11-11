"""Learned detector against the colour-threshold baseline on full frames.

Three appearance conditions: nominal (what the threshold was tuned for), the randomized
training distribution, and a harder one outside it (1.5x the ranges, distractor always on).

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python scripts/eval_detector.py
"""
import csv
import time
from pathlib import Path

import mujoco
import numpy as np

from g1catch import vision as vis
from g1catch.detector import CnnDetector, roi_around
from g1catch.render import Appearance, StereoRenderer
from g1catch.sim import ARM_DOF, READY_POSTURE, CatchSim
from train_detector import MAX_BLUR_PX, centre_visible, sample_ball_in_view

RESULTS = Path(__file__).resolve().parents[1] / "results"
SCENES = 400
OUTLIER_PX = 3.0
RANGE_BINS_M = ((0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.5))


def look_for(condition: str, rng) -> Appearance:
    if condition == "nominal":
        return Appearance.nominal()
    if condition == "randomized":
        return Appearance.random(rng, 1.0)
    look = Appearance.random(rng, 1.5)
    look.distractor = True
    return look


def main():
    sim = CatchSim()
    ren = StereoRenderer(sim.model)
    intr = vis.Intrinsics.from_fovy(sim.model.cam_fovy[0])
    detectors = {"cnn": CnnDetector(), "blob": vis.BlobDetector()}
    rows = []
    for condition in ("nominal", "randomized", "hard"):
        rng = np.random.default_rng(4242)
        for i in range(SCENES):
            d = sim.data
            mujoco.mj_resetData(sim.model, d)
            q = READY_POSTURE + rng.uniform(-0.4, 0.4, ARM_DOF)
            d.qpos[:ARM_DOF] = np.clip(q, sim.model.jnt_range[:ARM_DOF, 0], sim.model.jnt_range[:ARM_DOF, 1])
            mujoco.mj_forward(sim.model, d)
            poses = vis.camera_poses(sim.model, d)
            ball = sample_ball_in_view(rng, intr, poses[0])
            absent = rng.random() < 0.15           # some frames have no ball at all
            d.qpos[ARM_DOF:ARM_DOF + 3] = [0.0, 0.0, -5.0] if absent else ball
            mujoco.mj_forward(sim.model, d)
            px, depth = vis.project(intr, poses[0], ball)
            angle = rng.uniform(0, 2 * np.pi)
            shift = rng.uniform(0, MAX_BLUR_PX) * np.array([np.cos(angle), np.sin(angle)])
            image = ren.render(d, look_for(condition, rng), rng, [(px, shift), None])[0]
            visible = (not absent) and ball[2] > 0.04 and centre_visible(sim, poses[0].position, ball)
            for name, det in detectors.items():
                out = det.detect(image)
                err = float(np.linalg.norm(out.pixel - px)) if (out is not None and visible) else float("nan")
                rows.append({"condition": condition, "detector": name, "visible": int(visible), "range_m": float(depth),
                             "detected": int(out is not None), "error_px": err,
                             "du": float(out.pixel[0] - px[0]) if out is not None and visible else float("nan"),
                             "dv": float(out.pixel[1] - px[1]) if out is not None and visible else float("nan")})
    with open(RESULTS / "detector_full_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for condition in ("nominal", "randomized", "hard"):
        for name in detectors:
            r = [x for x in rows if x["condition"] == condition and x["detector"] == name]
            vis_rows = [x for x in r if x["visible"]]
            hid = [x for x in r if not x["visible"]]
            errs = np.array([x["error_px"] for x in vis_rows if x["detected"]])
            good = errs[errs < OUTLIER_PX]
            s = {"condition": condition, "detector": name, "visible_frames": len(vis_rows),
                 "found": float(np.mean([x["detected"] for x in vis_rows])),
                 "found_within_3px": float(np.sum(errs < OUTLIER_PX) / len(vis_rows)),
                 "median_px": float(np.median(good)) if len(good) else float("nan"),
                 "rms_axis_px": float(np.sqrt(np.mean(good**2) / 2)) if len(good) else float("nan"),
                 "false_detections": float(np.mean([x["detected"] for x in hid])) if hid else float("nan")}
            summary.append(s)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    with open(RESULTS / "detector_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print("CNN error by range, randomized appearance:")
    for lo, hi in RANGE_BINS_M:
        e = np.array([x["error_px"] for x in rows if x["condition"] == "randomized" and x["detector"] == "cnn"
                      and x["visible"] and x["detected"] and lo <= x["range_m"] < hi])
        if len(e):
            print(f"  {lo:.1f} to {hi:.1f} m: n {len(e)}, median {np.median(e):.2f} px, within 3 px {np.mean(e < OUTLIER_PX):.2f}")

    image = ren.render(sim.data, Appearance.nominal(), np.random.default_rng(0))[0]
    cnn = detectors["cnn"]
    for label, roi in (("full frame", None), ("96 px window", roi_around(np.array([160.0, 120.0]), intr.width, intr.height))):
        t0 = time.perf_counter()
        for _ in range(30):
            cnn.detect(image, roi)
        print(f"CNN inference, {label}: {1e3 * (time.perf_counter() - t0) / 30:.1f} ms")
    ren.close()


if __name__ == "__main__":
    main()
