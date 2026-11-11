"""Train and test the catchability classifier, then use it to refuse hopeless throws.

Throws are aimed at a box much larger than the arm's workspace. The classifier sees only what
the planner knows at commit time. It is compared with gradient-boosted trees on the same features
and then placed in the loop as a gate.

    python scripts/catchability.py
"""
import csv
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from g1catch.catchability import FEATURE_NAMES, CatchGate, feature_vector
from g1catch.experiment import RESULTS, Trial, run_trials, write_rows
from g1catch.runtime import RuntimeConfig
from g1catch.sim import WIDE_AIM_HI, WIDE_AIM_LO, ThrowDistribution
from plotstyle import MUTED, SERIES, apply_style

TRAIN_THROWS = 400
TEST_THROWS = 300
GATE_THROWS = 200
BASE = {"time_to_go", "slack", "reach", "rel_x", "rel_y", "rel_z", "sigma", "speed", "descent", "travel"}


def committed(rows):
    """Feature dicts and outcomes for the throws the planner committed to."""
    feats, caught = [], []
    for r in rows:
        if r["committed"]:
            feats.append({k[2:]: v for k, v in r.items() if k.startswith("f_")})
            caught.append(r["caught"])
    return feats, np.array(caught)


def reliability(prob, outcome, bins=8):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi + (hi == 1.0))
        if m.sum() >= 5:
            out.append((prob[m].mean(), outcome[m].mean(), int(m.sum())))
    return np.array(out)


def main():
    wide = ThrowDistribution(aim_lo=WIDE_AIM_LO, aim_hi=WIDE_AIM_HI)
    half = TRAIN_THROWS // 2
    data = run_trials([Trial("train a", RuntimeConfig(), wide, half, 2000), Trial("train b", RuntimeConfig(), wide, half, 2001),
                       Trial("test a", RuntimeConfig(), wide, TEST_THROWS // 2, 2100),
                       Trial("test b", RuntimeConfig(), wide, TEST_THROWS // 2, 2101)])
    train_rows = data["train a"] + data["train b"]
    test_rows = data["test a"] + data["test b"]
    write_rows(RESULTS / "catchability_throws.csv", train_rows + test_rows)
    f_train, y_train = committed(train_rows)
    f_test, y_test = committed(test_rows)
    print(f"train: {len(train_rows)} throws, {len(f_train)} committed, {int(y_train.sum())} caught")
    print(f"test:  {len(test_rows)} throws, {len(f_test)} committed, {int(y_test.sum())} caught")

    gate = CatchGate.fit(f_train, y_train)
    p_test = np.array([gate.probability(f) for f in f_test])
    x_train = np.array([feature_vector(f) for f in f_train])
    x_test = np.array([feature_vector(f) for f in f_test])
    trees = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0).fit(x_train, y_train)
    p_trees = trees.predict_proba(x_test)[:, 1]
    base_rate = y_train.mean()
    rows = []
    for name, p in (("logistic regression", p_test), ("boosted trees", p_trees), ("base rate only", np.full(len(y_test), base_rate))):
        auc = roc_auc_score(y_test, p) if name != "base rate only" else 0.5
        acc = np.mean((p >= 0.5) == y_test)
        rows.append({"model": name, "auc": auc, "brier": brier_score_loss(y_test, p), "accuracy": acc})
        print(f"{name:20s} AUC {auc:.3f}  Brier {brier_score_loss(y_test, p):.3f}  accuracy {acc:.3f}")
    with open(RESULTS / "catchability_models.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    order = np.argsort(-np.abs(gate.weights))
    print("largest standardised weights:", ", ".join(f"{FEATURE_NAMES[i]} {gate.weights[i]:+.2f}" for i in order[:6]))
    with open(RESULTS / "catchability_weights.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "standardised_weight"])
        w.writerows((FEATURE_NAMES[i], gate.weights[i]) for i in order)
    with open(RESULTS / "catch_gate.pkl", "wb") as f:
        pickle.dump(gate, f)

    rel = reliability(p_test, y_test)
    apply_style()
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax.plot(rel[:, 0], rel[:, 1], color=SERIES[0], lw=2, marker="o", ms=5)
    for x, y, n in rel:
        ax.annotate(f"{int(n)}", (x, y), textcoords="offset points", xytext=(5, -10), fontsize=7, color=MUTED)
    ax.set_xlabel("predicted probability of a catch")
    ax.set_ylabel("fraction caught")
    ax.set_title("Catchability classifier on held-out throws\n(numbers: throws per bin)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "catchability_reliability.png", dpi=160)

    # In the loop on fresh throws: same seeds with and without the gate.
    loop = run_trials([Trial("no gate", RuntimeConfig(), wide, GATE_THROWS, 2200),
                       Trial("gate at 0.5", RuntimeConfig(), wide, GATE_THROWS, 2200, catch_gate=gate)])
    summary = []
    for name, r in loop.items():
        attempts = sum(x["committed"] for x in r)
        caught = sum(x["caught"] for x in r)
        s = {"condition": name, "throws": len(r), "attempts": attempts, "caught": caught,
             "failed_attempts": attempts - caught, "self_collisions": sum(x["self_collided"] for x in r)}
        summary.append(s)
        print(s)
    write_rows(RESULTS / "catchability_gate.csv", summary)


if __name__ == "__main__":
    main()
