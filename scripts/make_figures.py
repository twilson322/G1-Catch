"""Figures for the README, drawn from the CSV files in results/.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python scripts/make_figures.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import mujoco
import numpy as np

from g1catch.experiment import Trial, run_trial
from g1catch.runtime import RuntimeConfig
from g1catch.sim import ThrowDistribution
from plotstyle import INK, MUTED, SERIES, SURFACE, apply_style

RESULTS = Path(__file__).resolve().parents[1] / "results"


def read(name):
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))


def num(row, key):
    try:
        return float(row[key])
    except (ValueError, KeyError):
        return float("nan")


def rate_bars(ax, rows, colors=None, label_width=None):
    """Horizontal bars of catch rate with 95 % intervals, one per row, labelled directly."""
    names = [r["trial"] for r in rows]
    rate = np.array([100 * num(r, "rate") for r in rows])
    lo = np.array([100 * num(r, "ci_lo") for r in rows])
    hi = np.array([100 * num(r, "ci_hi") for r in rows])
    y = np.arange(len(rows))[::-1]
    ax.barh(y, rate, height=0.62, color=colors or SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    ax.errorbar(rate, y, xerr=[rate - lo, hi - rate], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
    for yi, r, h in zip(y, rate, hi):
        ax.text(h + 1.5, yi, f"{r:.0f} %", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel("throws caught (%)")
    ax.grid(axis="y", visible=False)


def fig_ablations():
    rows = [r for r in read("exp_summary.csv") if r["group"] == "ablations"]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    colors = [SERIES[0]] + [MUTED] * (len(rows) - 1)
    rate_bars(ax, rows, colors)
    ax.set_title(f"Removing one component at a time ({rows[0]['n']} identical throws each, 95 % intervals)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_ablations.png", dpi=170)
    plt.close(fig)


def fig_sweeps():
    rows = read("exp_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)

    def line(ax, group, pick, x_of, color, label):
        sel = [r for r in rows if r["group"] == group and pick(r["trial"])]
        x = [x_of(r["trial"]) for r in sel]
        rate = np.array([100 * num(r, "rate") for r in sel])
        lo = np.array([100 * num(r, "ci_lo") for r in sel])
        hi = np.array([100 * num(r, "ci_hi") for r in sel])
        ax.errorbar(x, rate, yerr=[rate - lo, hi - rate], color=color, lw=2, marker="o", ms=5, capsize=2.5, label=label)

    first_number = lambda name: float(name.split()[0])
    line(axes[0], "latency", lambda n: "uncompensated" not in n, first_number, SERIES[0], "compensated")
    line(axes[0], "latency", lambda n: "uncompensated" in n, first_number, SERIES[1], "not compensated")
    axes[0].set_xlabel("camera-to-planner latency (ms)")
    axes[0].set_ylabel("throws caught (%)")
    axes[0].legend(frameon=False, fontsize=8)
    line(axes[1], "pixel_noise", lambda n: True, first_number, SERIES[0], None)
    axes[1].set_xscale("log")
    axes[1].set_xticks([0.25, 0.5, 1, 2])
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].minorticks_off()
    axes[1].set_xticklabels(["0.25", "0.5", "1", "2"])
    axes[1].set_xlabel("detector noise (pixels, 1 sigma)")
    e_of = lambda name: float(name.split("=")[1].split(",")[0])
    line(axes[2], "restitution", lambda n: "no velocity" not in n, e_of, SERIES[0], "velocity matched")
    line(axes[2], "restitution", lambda n: "no velocity" in n, e_of, SERIES[1], "stationary cup")
    axes[2].set_xlabel("restitution of the cup liner")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_sweeps.png", dpi=170)
    plt.close(fig)


def fig_tracking():
    """Tracker error and the filter's own uncertainty against time before the catch."""
    rows = run_trial(Trial("frames", RuntimeConfig(), ThrowDistribution(), 80, 77, keep_frames=True))
    togo, err, sigma = [], [], []
    for r in rows:
        if not r["committed"]:
            continue
        for f in r["_frames"]:
            if f["cameras"] and f["t"] < r["catch_time"]:
                togo.append(r["catch_time"] - f["t"])
                err.append(np.linalg.norm(f["est_p"] - f["true_p"]))
                sigma.append(f["sigma_now"])
    togo, err, sigma = map(np.array, (togo, err, sigma))
    edges = np.linspace(0.0, 0.7, 15)
    mid, med, p90, sig = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (togo >= lo) & (togo < hi)
        if m.sum() > 20:
            mid.append(0.5 * (lo + hi))
            med.append(100 * np.median(err[m]))
            p90.append(100 * np.percentile(err[m], 90))
            sig.append(100 * np.median(sigma[m]))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(mid, med, color=SERIES[0], lw=2, marker="o", ms=4, label="position error, median")
    ax.plot(mid, p90, color=SERIES[0], lw=1.2, ls="--", label="position error, 90th percentile")
    ax.plot(mid, sig, color=SERIES[1], lw=2, marker="s", ms=4, label="filter's own 1-sigma")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("time before the catch (s)")
    ax.set_ylabel("ball position (cm)")
    ax.set_title("Stereo tracking gets better as the ball gets closer", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_tracking.png", dpi=170)
    plt.close(fig)
    with open(RESULTS / "tracking_vs_time.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_before_catch_s", "median_error_cm", "p90_error_cm", "median_filter_sigma_cm"])
        w.writerows(zip(mid, med, p90, sig))


def fig_detector():
    rows = read("detector_summary.csv")
    conditions = ["nominal", "randomized", "hard"]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    width = 0.36
    for i, (det, label) in enumerate((("cnn", "CNN heatmap"), ("blob", "colour threshold"))):
        vals = [100 * num(next(r for r in rows if r["condition"] == c and r["detector"] == det), "found_within_3px") for c in conditions]
        x = np.arange(len(conditions)) + (i - 0.5) * (width + 0.03)
        ax.bar(x, vals, width=width, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(["nominal look", "training randomization", "beyond training"])
    ax.set_ylabel("visible balls found within 3 px (%)")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_detector.png", dpi=170)
    plt.close(fig)


def fig_learning():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    d = read("dagger.csv")
    mpc = next(r for r in d if r["round"] == "mpc")
    rounds = [r for r in d if r["round"] != "mpc" and "no barrier" not in r["round"]]
    x = np.arange(len(rounds))
    rate = np.array([100 * num(r, "rate") for r in rounds])
    lo = np.array([100 * num(r, "ci_lo") for r in rounds])
    hi = np.array([100 * num(r, "ci_hi") for r in rounds])
    axes[0].axhspan(100 * num(mpc, "ci_lo"), 100 * num(mpc, "ci_hi"), color=MUTED, alpha=0.15, lw=0)
    axes[0].axhline(100 * num(mpc, "rate"), color=MUTED, lw=1.5, ls="--")
    axes[0].text(len(rounds) - 1, 100 * num(mpc, "rate") + 1.5, "MPC", ha="right", fontsize=8, color=MUTED)
    axes[0].errorbar(x, rate, yerr=[rate - lo, hi - rate], color=SERIES[0], lw=2, marker="o", ms=5, capsize=2.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["cloning"] + [f"DAgger {i}" for i in range(1, len(rounds))], fontsize=8)
    axes[0].set_ylabel("throws caught (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Network distilled from the MPC, driving the arm", fontsize=9)
    p = [r for r in read("ppo_residual.csv") if r["iteration"] != "-1"]
    axes[1].plot([int(r["iteration"]) + 1 for r in p], [100 * num(r, "train_catch_rate") for r in p], color=SERIES[0], lw=2,
                 marker="o", ms=4, label="training throws (exploring)")
    ev = [r for r in read("ppo_residual.csv") if r["eval_rate"] not in ("", "nan")]
    axes[1].plot([int(r["iteration"]) + 1 for r in ev], [100 * num(r, "eval_rate") for r in ev], color=SERIES[1], lw=2, marker="s",
                 ms=5, label="evaluation throws (iteration 0: no correction)")
    axes[1].set_xlabel("PPO iteration (160 throws each)")
    axes[1].set_ylabel("throws caught (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("PPO correction to the catch point", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_learning.png", dpi=170)
    plt.close(fig)


def fig_filmstrip():
    """One catch seen from outside and from the robot's left eye."""
    from PIL import Image

    from g1catch.detector import CnnDetector
    from g1catch.render import StereoRenderer
    from g1catch.runtime import CatchRunner
    from g1catch.sensing import RenderedSensor
    from g1catch.sim import CatchSim
    from g1catch.vision import Intrinsics

    sim = CatchSim()
    intr = Intrinsics.from_fovy(sim.model.cam_fovy[0])
    sensor = RenderedSensor(intr, StereoRenderer(sim.model), CnnDetector(), np.random.default_rng(0))
    runner = CatchRunner(sim, sensor, RuntimeConfig())
    outside = mujoco.Renderer(sim.model, 360, 480)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.9, -0.1, 1.15]
    camera.distance, camera.azimuth, camera.elevation = 3.1, 62, -8
    shots, eye = [], []
    wanted = [0.20, 0.45, 0.62, 0.70, 0.95]
    original_step = sim.step

    def step(tau):
        original_step(tau)
        if wanted and sim.time >= wanted[0]:
            wanted.pop(0)
            outside.update_scene(sim.data, camera)
            shots.append(outside.render().copy())
            if sensor.last_images is not None:
                eye.append(sensor.last_images[0].copy())
    sim.step = step
    rng = np.random.default_rng(21)
    dist = ThrowDistribution(flight_time_s=(0.70, 0.72))
    for _ in range(20):
        shots.clear(); eye.clear(); wanted[:] = [0.20, 0.45, 0.62, 0.70, 0.95]
        if runner.run(dist.sample(rng)).caught:
            break
    strip = np.concatenate(shots, axis=1)
    Image.fromarray(strip).save(RESULTS / "fig_catch.png")
    if eye:
        Image.fromarray(np.concatenate(eye[:4], axis=1)).resize((4 * 400, 300)).save(RESULTS / "fig_head_camera.png")


if __name__ == "__main__":
    apply_style()
    import sys
    wanted = sys.argv[1:]
    for make in (fig_ablations, fig_sweeps, fig_tracking, fig_detector, fig_learning, fig_filmstrip):
        if wanted and make.__name__ not in wanted:
            continue
        try:
            make()
            print("ok ", make.__name__)
        except (FileNotFoundError, StopIteration) as exc:
            print("skip", make.__name__, "-", exc)
