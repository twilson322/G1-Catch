"""Render a randomized dataset of ball crops and train the heatmap detector.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python scripts/train_detector.py

Outputs results/detector.pt, results/detector_training.csv and results/detector_eval.csv.
"""
import argparse
import csv
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from g1catch import vision as vis
from g1catch.detector import CROP_PX, HeatmapNet, gaussian_target, peak_centroid, DETECTION_THRESHOLD
from g1catch.render import Appearance, StereoRenderer
from g1catch.sim import ARM_DOF, READY_POSTURE, CatchSim

RESULTS = Path(__file__).resolve().parents[1] / "results"
MIN_RANGE_M, MAX_RANGE_M = 0.30, 4.5
MAX_BLUR_PX = 6.0
POSTURE_SPREAD_RAD = 0.5
POSITIVE_WEIGHT = 30.0     # heatmap pixels near a centre are about 1 in 1000


def sample_ball_in_view(rng, intr, pose):
    """A world point inside the camera frustum, uniform in the image.

    Half the samples are uniform in range and half uniform in inverse range. Inverse range alone
    puts nine tenths of the balls within 2 m, and the throws start beyond 3 m where the ball is
    two pixels wide.
    """
    if rng.random() < 0.5:
        depth = rng.uniform(MIN_RANGE_M, MAX_RANGE_M)
    else:
        depth = 1.0 / rng.uniform(1.0 / MAX_RANGE_M, 1.0 / MIN_RANGE_M)
    u = rng.uniform(4, intr.width - 5)
    v = rng.uniform(4, intr.height - 5)
    pc = np.array([(u - intr.cx) / intr.focal_px * depth, -(v - intr.cy) / intr.focal_px * depth, -depth])
    return pose.position + pose.rotation @ pc


def centre_visible(sim, cam_pos, ball_pos) -> bool:
    """True if nothing but the ball lies on the ray from the camera to the ball's centre."""
    direction = ball_pos - cam_pos
    dist = np.linalg.norm(direction)
    geomid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(sim.model, sim.data, cam_pos, direction / dist, None, 1, -1, geomid)
    return hit >= 0 and geomid[0] == sim.model.geom("ball_geom").id


def make_dataset(n: int, seed: int, strength: float = 1.0):
    """Returns crops (n, 96, 96, 3) uint8, centre pixels in the crop (n, 2), visibility (n,), range (n,)."""
    rng = np.random.default_rng(seed)
    sim = CatchSim()
    ren = StereoRenderer(sim.model)
    intr = vis.Intrinsics.from_fovy(sim.model.cam_fovy[0])
    crops = np.empty((n, CROP_PX, CROP_PX, 3), np.uint8)
    centres = np.zeros((n, 2), np.float32)
    visible = np.zeros(n, bool)
    ranges = np.zeros(n, np.float32)
    i = 0
    while i < n:
        d = sim.data
        mujoco.mj_resetData(sim.model, d)
        q = READY_POSTURE + rng.uniform(-POSTURE_SPREAD_RAD, POSTURE_SPREAD_RAD, ARM_DOF)
        d.qpos[:ARM_DOF] = np.clip(q, sim.joint_lo, sim.joint_hi)
        mujoco.mj_forward(sim.model, d)
        poses = vis.camera_poses(sim.model, d)
        cam = int(rng.integers(2))
        ball = sample_ball_in_view(rng, intr, poses[cam])
        if ball[2] < 0.04:
            continue
        d.qpos[ARM_DOF:ARM_DOF + 3] = ball
        mujoco.mj_forward(sim.model, d)
        look = Appearance.random(rng, strength)
        blur = [None, None]
        px, depth = vis.project(intr, poses[cam], ball)
        if rng.random() < 0.6:
            angle = rng.uniform(0, 2 * np.pi)
            blur[cam] = (px, rng.uniform(0, MAX_BLUR_PX) * np.array([np.cos(angle), np.sin(angle)]))
        image = ren.render(d, look, rng, blur)[cam]
        seen = centre_visible(sim, poses[cam].position, ball)
        # One crop that contains the ball at a random offset and one random crop elsewhere.
        for want_ball in (True, False):
            if i >= n:
                break
            if want_ball:
                u0 = int(np.clip(px[0] - rng.uniform(8, CROP_PX - 8), 0, intr.width - CROP_PX))
                v0 = int(np.clip(px[1] - rng.uniform(8, CROP_PX - 8), 0, intr.height - CROP_PX))
            else:
                u0 = int(rng.integers(0, intr.width - CROP_PX + 1))
                v0 = int(rng.integers(0, intr.height - CROP_PX + 1))
            local = px - np.array([u0, v0])
            inside = bool(0 <= local[0] <= CROP_PX - 1 and 0 <= local[1] <= CROP_PX - 1)
            crops[i] = image[v0:v0 + CROP_PX, u0:u0 + CROP_PX]
            centres[i] = local
            visible[i] = seen and inside
            ranges[i] = depth
            i += 1
    ren.close()
    return crops, centres, visible, ranges


def evaluate(net, crops, centres, visible, ranges):
    """Per-sample pixel error for visible balls, miss flags, and false-positive flags."""
    net.eval()
    rows = []
    with torch.no_grad():
        for s in range(0, len(crops), 256):
            x = torch.from_numpy(crops[s:s + 256]).permute(0, 3, 1, 2).float() / 255.0
            prob = torch.sigmoid(net(x))[:, 0].numpy()
            for j in range(len(prob)):
                c, peak = peak_centroid(prob[j])
                k = s + j
                detected = peak >= DETECTION_THRESHOLD
                err = float(np.linalg.norm(c - centres[k])) if visible[k] else float("nan")
                rows.append((bool(visible[k]), float(ranges[k]), detected, err, c[0] - centres[k][0], c[1] - centres[k][1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=14000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch threads, 0 leaves the default")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    RESULTS.mkdir(exist_ok=True)

    t0 = time.time()
    cache = RESULTS / "detector_data.npz"
    if cache.exists():
        z = np.load(cache)
        train = (z["tc"], z["tp"], z["tv"], z["tr"])
        val = (z["vc"], z["vp"], z["vv"], z["vr"])
    else:
        train = make_dataset(args.train, seed=args.seed)
        val = make_dataset(args.val, seed=args.seed + 10_000)
        np.savez_compressed(cache, tc=train[0], tp=train[1], tv=train[2], tr=train[3],
                            vc=val[0], vp=val[1], vv=val[2], vr=val[3])
    print(f"dataset ready in {time.time() - t0:.0f} s, {train[2].mean():.2f} of training crops show the ball", flush=True)

    net = HeatmapNet()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    steps_per_epoch = len(train[0]) // 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=args.epochs * steps_per_epoch)
    rng = np.random.default_rng(args.seed)
    log = []
    for epoch in range(args.epochs):
        net.train()
        order = rng.permutation(len(train[0]))
        total = 0.0
        for b in range(steps_per_epoch):
            idx = order[b * 64:(b + 1) * 64]
            x = torch.from_numpy(train[0][idx]).permute(0, 3, 1, 2).float() / 255.0
            y = torch.from_numpy(gaussian_target(train[1][idx], train[2][idx]))
            logits = net(x)[:, 0]
            weight = 1.0 + (POSITIVE_WEIGHT - 1.0) * y
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        rows = evaluate(net, *val)
        errs = np.array([r[3] for r in rows if r[0] and r[2]])
        miss = np.mean([not r[2] for r in rows if r[0]])
        false_pos = np.mean([r[2] for r in rows if not r[0]])
        log.append((epoch, total / steps_per_epoch, float(np.median(errs)), float(np.percentile(errs, 95)), miss, false_pos))
        print("epoch %d loss %.4f  median %.2f px  p95 %.2f px  miss %.3f  false+ %.3f  (%.0f s)" % (*log[-1], time.time() - t0), flush=True)

    torch.save(net.state_dict(), RESULTS / "detector.pt")
    with open(RESULTS / "detector_training.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_median_px", "val_p95_px", "val_miss_rate", "val_false_positive_rate"])
        w.writerows(log)
    with open(RESULTS / "detector_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["visible", "range_m", "detected", "error_px", "du_px", "dv_px"])
        w.writerows(evaluate(net, *val))


if __name__ == "__main__":
    main()
