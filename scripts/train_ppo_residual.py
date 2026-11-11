"""Train the PPO correction to the catch point and compare it with the model-based pipeline.

    python scripts/train_ppo_residual.py
"""
import csv
import sys

import numpy as np
import torch

from g1catch.experiment import RESULTS, Trial, catch_summary, run_trials
from g1catch.ppo import InterceptAdjuster, ResidualPolicy, ppo_update, reward
from g1catch.runtime import RuntimeConfig
from g1catch.sim import ThrowDistribution

ITERATIONS = 8
THROWS_PER_ITERATION = 160
EVAL_THROWS = 160
EVAL_SEED = 4900
LEARNING_RATE = 3e-4
HEAD_PITCH_ERROR_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
TAG = f"_pitch{HEAD_PITCH_ERROR_DEG:g}" if HEAD_PITCH_ERROR_DEG else ""


def config() -> RuntimeConfig:
    """The robot the correction is trained on. With a head pitch error every forecast is
    offset the same way, which is a systematic error a learned correction could remove."""
    cfg = RuntimeConfig()
    cfg.head_pitch_error_deg = HEAD_PITCH_ERROR_DEG
    return cfg


def snapshot(policy):
    return {k: v.clone() for k, v in policy.state_dict().items()}


def evaluate(policy, dist):
    adjust = InterceptAdjuster(snapshot(policy), seed=0, explore=False)
    half = EVAL_THROWS // 2
    out = run_trials([Trial("a", config(), dist, half, EVAL_SEED, intercept_adjust=adjust),
                      Trial("b", config(), dist, half, EVAL_SEED + 1, intercept_adjust=adjust)])
    return out["a"] + out["b"]


def main():
    if 0.0 < HEAD_PITCH_ERROR_DEG < 1.0:
        raise SystemExit("a head pitch error below 1 degree does not change the catch rate (scripts/model_error.py)")
    torch.manual_seed(0)
    policy = ResidualPolicy()
    optimiser = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    dist = ThrowDistribution()
    half = EVAL_THROWS // 2
    base = run_trials([Trial("a", config(), dist, half, EVAL_SEED), Trial("b", config(), dist, half, EVAL_SEED + 1)])
    base_rows = base["a"] + base["b"]
    s0 = catch_summary(base_rows)
    print(f"model-based pipeline on the evaluation throws: {s0['caught']}/{s0['n']} = {100 * s0['rate']:.1f} %", flush=True)
    log = [{"iteration": -1, "train_reward": float("nan"), "train_catch_rate": float("nan"), "eval_caught": s0["caught"],
            "eval_n": s0["n"], "eval_rate": s0["rate"], "action_std": float("nan")}]
    for it in range(ITERATIONS):
        state = snapshot(policy)
        trials = [Trial(f"w{w}", config(), dist, THROWS_PER_ITERATION // 2, 5000 + 10 * it + w,
                        intercept_adjust=InterceptAdjuster(state, seed=7000 + 10 * it + w, explore=True)) for w in range(2)]
        out = run_trials(trials)
        rows = [r for t in trials for r in out[t.name] if "_decision" in r]
        obs = np.array([r["_decision"][0] for r in rows])
        actions = np.array([r["_decision"][1] for r in rows])
        log_probs = np.array([r["_decision"][2] for r in rows], dtype=np.float32)
        rewards = np.array([reward(r) for r in rows], dtype=np.float32)
        ppo_update(policy, optimiser, obs, actions, log_probs, rewards, seed=it)
        entry = {"iteration": it, "train_reward": float(rewards.mean()), "train_catch_rate": float(np.mean([r["caught"] for r in rows])),
                 "eval_caught": "", "eval_n": "", "eval_rate": "", "action_std": float(policy.log_std.exp().mean())}
        if it % 4 == 3 or it == ITERATIONS - 1:
            s = catch_summary(evaluate(policy, dist))
            entry.update({"eval_caught": s["caught"], "eval_n": s["n"], "eval_rate": s["rate"]})
        log.append(entry)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items()}, flush=True)
    torch.save(policy.state_dict(), RESULTS / f"ppo_residual{TAG}.pt")
    with open(RESULTS / f"ppo_residual{TAG}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)

    final_rows = evaluate(policy, dist)
    only_base = sum(a["caught"] and not b["caught"] for a, b in zip(base_rows, final_rows))
    only_ppo = sum(b["caught"] and not a["caught"] for a, b in zip(base_rows, final_rows))
    from scipy.stats import binomtest
    p = binomtest(only_base, only_base + only_ppo, 0.5).pvalue if only_base + only_ppo else 1.0
    s = catch_summary(final_rows)
    print(f"with the learned correction: {s['caught']}/{s['n']} = {100 * s['rate']:.1f} %. "
          f"Throws only the baseline caught: {only_base}, only the correction caught: {only_ppo}, McNemar p = {p:.3f}")
    with open(RESULTS / f"ppo_residual_final{TAG}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "caught", "n", "only_this_caught", "mcnemar_p"])
        w.writerow(["model-based", s0["caught"], s0["n"], only_base, p])
        w.writerow(["with PPO correction", s["caught"], s["n"], only_ppo, p])


if __name__ == "__main__":
    main()
