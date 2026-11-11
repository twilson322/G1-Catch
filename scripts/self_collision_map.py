"""Where does the arm hit the body? Sampled joint configurations against MuJoCo's collision meshes.

This is the evidence for the planning limit on shoulder roll in g1catch/sim.py.

    python scripts/self_collision_map.py
"""
import mujoco
import numpy as np

from g1catch.sim import ARM_DOF, SHOULDER_ROLL_INDEX, CatchSim

SAMPLES = 20000


def main():
    sim = CatchSim()
    m, d = sim.model, sim.data
    rng = np.random.default_rng(0)
    lo, hi = m.jnt_range[:ARM_DOF, 0].copy(), m.jnt_range[:ARM_DOF, 1].copy()
    lo[3], hi[3] = -2.0, 0.6            # shoulder pitch range used for catching
    lo[SHOULDER_ROLL_INDEX], hi[SHOULDER_ROLL_INDEX] = -1.2, 0.8
    rolls, upper_arm_hits, any_hits = [], [], []
    pairs = {}
    for _ in range(SAMPLES):
        q = rng.uniform(lo, hi)
        d.qpos[:ARM_DOF] = q
        d.qpos[ARM_DOF:ARM_DOF + 3] = [5.0, 5.0, 1.0]     # ball out of the way
        mujoco.mj_forward(m, d)
        hit = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < 0:
                hit.add((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name))
        for pair in hit:
            pairs[pair] = pairs.get(pair, 0) + 1
        rolls.append(q[SHOULDER_ROLL_INDEX])
        upper_arm_hits.append(("torso_link", "right_shoulder_yaw_link") in hit)
        any_hits.append(bool(hit))
    rolls, upper_arm_hits, any_hits = map(np.array, (rolls, upper_arm_hits, any_hits))
    print("most frequent colliding pairs:")
    for pair, count in sorted(pairs.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {pair[0]:28s} {pair[1]:28s} {count / SAMPLES:.3f}")
    print("upper arm against torso, by shoulder roll (positive rolls the arm inward):")
    for start in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
        sel = (rolls >= start) & (rolls < start + 0.2)
        print(f"  {start:+.1f} to {start + 0.2:+.1f} rad: {upper_arm_hits[sel].mean():.2f}")
    outside = rolls < sim.joint_hi[SHOULDER_ROLL_INDEX]
    print(f"with the planning limit (roll < {sim.joint_hi[SHOULDER_ROLL_INDEX]:.2f}): upper arm against torso "
          f"{upper_arm_hits[outside].mean():.3f}, any self-collision {any_hits[outside].mean():.3f}")


if __name__ == "__main__":
    main()
