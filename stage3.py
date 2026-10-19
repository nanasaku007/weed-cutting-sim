"""Paired teacher evaluation and versioned, balanced demonstration collection."""
import argparse
from collections import Counter
from hashlib import sha256
import itertools
from pathlib import Path
import time

import mujoco
import numpy as np

from environment import CuttingEnv, OBS_V2_NAMES, ROOT
from expert import blocked_positions
from run import provenance, save_json

CONDITIONS = list(itertools.product(("slender", "broad", "tall"), (0, 10, 20)))
SCHEMA = "state30_v2"


def initialize(env, seed, recovery=False, ood=False):
    obs = env.reset(seed, ood)
    if not recovery:
        return obs
    # A labelled initial condition, never a teleport within a demonstration.
    # The same deterministic perturbation is used for both teachers.
    rng = np.random.default_rng(seed + 900000)
    offset = np.array([-rng.uniform(0.018, 0.035), rng.uniform(-0.008, 0.008), rng.uniform(-0.012, 0.012)])
    position = env.target + offset
    q, residual = env.solve_ik(position, env.rest_q, iterations=150)
    if residual > 0.001:
        raise ValueError("recovery_ik_failed")
    env.data.qpos[env.arm_q] = q
    env.data.ctrl[:7] = q
    env.data.qpos[env.scissor_q] = 0.
    env.data.ctrl[7] = 0.
    env.previous_angle = 0.
    mujoco.mj_forward(env.model, env.data)
    # Reject invalid starts, retain them in the attempt manifest; do not retry seeds.
    protected = np.arange(3) != env.target_id
    if blocked_positions(env.tip, env.roots[protected],
                         env.stems[protected] + env.upper_length * env.stem_direction, 0.)[0]:
        raise ValueError("recovery_start_blocked")
    env._events()
    if env.protected_touched or env.terrain_contact or env.cut.any():
        raise ValueError("recovery_start_invalid")
    return env.observe()


def outcome(result, last_phase):
    if result["success"]:
        return "success"
    if result["wrong_cuts"]:
        return "wrong_cut"
    if result["protected_intrusions"]:
        return "protected_intrusion"
    if result["tool_terrain_contact"]:
        return "terrain_contact"
    return "no_safe_route" if last_phase.startswith("blocked") else "timeout"


def episode(env, seed, recovery=False, ood=False, record=False):
    try:
        obs = initialize(env, seed, recovery, ood)
    except ValueError as error:
        return {"seed": seed, "recovery": recovery, "success": False,
                "outcome": str(error), "environment": env.configuration()}, []
    frames, phases = [], Counter()
    reopened = False
    done = False
    while not done:
        action = env.expert()
        phase = env.expert_status if env.teacher == "waypoint_v2" else "direct"
        phases[phase] += 1
        nxt, done, result = env.step(action)
        reopened |= recovery and not result["target_cut"] and env.data.qpos[env.scissor_q] > 0.12
        if record:
            frames.append((obs, action, nxt, phase))
        obs = nxt
    return {**result, "outcome": outcome(result, phase), "phases": dict(phases),
            "recovery": recovery, "reopened_before_cut": bool(reopened),
            "environment": env.configuration()}, frames


def metadata():
    report = {**provenance(), "observations": OBS_V2_NAMES, "observation_schema": SCHEMA}
    report["source_sha256"]["stage3.py"] = sha256((ROOT / "stage3.py").read_bytes()).hexdigest()
    return report


def evaluation(args):
    report = {**metadata(), "seed_start": args.seed, "count_per_condition": args.episodes,
              "recovery": args.recovery, "ood": args.ood, "conditions": []}
    start = time.monotonic()
    for plant, tilt in CONDITIONS:
        for teacher in ("direct_v1", "waypoint_v2"):
            env = CuttingEnv(plant=plant, tilt=tilt, geometry="stem_segment_v1",
                             teacher=teacher, observation_schema=SCHEMA)
            try:
                rows = []
                for i in range(args.episodes):
                    env.azimuth = float((i % 4) * 90)
                    result, _ = episode(env, args.seed + i, args.recovery, args.ood)
                    rows.append(result)
                condition = {"plant": plant, "tilt": tilt, "teacher": teacher,
                             "success_count": sum(r["success"] for r in rows),
                             "outcomes": dict(Counter(r["outcome"] for r in rows)), "episodes": rows}
                report["conditions"].append(condition)
                print(plant, tilt, teacher, condition["outcomes"], flush=True)
            finally:
                env.close()
            # Retain partial reports if a long run is interrupted.
            report["elapsed_seconds"] = time.monotonic() - start
            save_json(args.out, report)


def collection(args):
    rows, frames, ids, seeds, conditions, recovery_flags, phases = [], [], [], [], [], [], []
    for condition_id, (plant, tilt) in enumerate(CONDITIONS):
        env = CuttingEnv(plant=plant, tilt=tilt, geometry="stem_segment_v1",
                         teacher="waypoint_v2", observation_schema=SCHEMA)
        try:
            for i in range(args.episodes):
                episode_id = condition_id * args.episodes + i
                seed = args.seed + episode_id
                env.azimuth = float((i % 4) * 90)
                recovery = i % 5 == 0
                result, data = episode(env, seed, recovery, record=True)
                result.update(episode_id=episode_id, condition_id=condition_id,
                              split="validation" if i % 25 in (0, 4, 9, 14, 19) else "train")
                rows.append(result)
                if result["success"]:
                    frames.extend([f[:3] for f in data])
                    phases.extend([f[3] for f in data])
                    ids.extend([episode_id] * len(data)); seeds.extend([seed] * len(data))
                    conditions.extend([condition_id] * len(data)); recovery_flags.extend([recovery] * len(data))
            selected = rows[-args.episodes:]
            print(plant, tilt, dict(Counter(r['outcome'] for r in selected)), flush=True)
        finally:
            env.close()
        save_json(Path(args.out).with_suffix(".attempts.json"), {**metadata(), "episodes": rows})
    if not frames:
        raise RuntimeError("No successful demonstrations")
    observations, actions, following = map(np.asarray, zip(*frames))
    val_ids = [r["episode_id"] for r in rows if r["success"] and r["split"] == "validation"]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, observation=observations, action=actions, next_observation=following,
                        episode_id=ids, seed=seeds, condition_id=conditions, recovery=recovery_flags,
                        teacher_phase=phases, observation_schema=SCHEMA, validation_episode_ids=val_ids)
    report = {**metadata(), "dataset_version": "waypoint_state30_v1", "seed_start": args.seed,
              "attempt_count": len(rows), "success_count": sum(r["success"] for r in rows),
              "frames": len(frames), "episodes": rows, "validation_episode_ids": val_ids,
              "dataset_sha256": sha256(path.read_bytes()).hexdigest(),
              "split_rule": "per condition: i%25 in (0,4,9,14,19) validation; otherwise train; no repeated seeds"}
    save_json(path.with_suffix(".json"), report)
    print(f"Saved {path}: {report['success_count']}/{len(rows)} episodes, {len(frames)} frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "collect"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=30, help="attempts per shape/tilt condition")
    parser.add_argument("--out", required=True)
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--ood", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.command == "collect" and (args.recovery or args.ood):
        parser.error("collect uses a fixed 20% recovery mix and IID layouts")
    {"evaluate": evaluation, "collect": collection}[args.command](args)


if __name__ == "__main__":
    main()
