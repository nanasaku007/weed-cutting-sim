"""Paired closed-loop BC/teacher evaluation, without teacher corrections to BC."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import time

import numpy as np

from environment import CuttingEnv, DT, ROOT
from run import TreeBC, save_json
from stage3 import CONDITIONS, SCHEMA, initialize, metadata, outcome


def rollout(env, seed, policy=None, recovery=False):
    if policy is not None and policy.observation_schema != env.observation_schema:
        raise ValueError("Checkpoint and environment observation schemas differ")
    try:
        obs = initialize(env, seed, recovery)
    except ValueError as error:
        return {"seed": seed, "success": False, "outcome": str(error),
                "initialization_failed": True, "recovery": recovery,
                "environment": env.configuration()}, {}
    observations, actions, following, controls, ik_errors, phases = [], [], [], [], [], []
    reopened, done = False, False
    while not done:
        # BC only sees the observation. No expert/planner call in this branch.
        action = policy.predict(obs) if policy is not None else env.expert()
        phase = "bc" if policy is not None else env.expert_status
        nxt, done, result = env.step(action)
        observations.append(obs.copy()); actions.append(np.asarray(action).copy())
        following.append(nxt.copy()); controls.append(env.data.ctrl.copy())
        ik_errors.append(env.last_ik_error); phases.append(phase)
        reopened |= recovery and not result["target_cut"] and env.data.qpos[env.scissor_q] > 0.12
        obs = nxt
    trace = {"observation": np.asarray(observations), "action": np.asarray(actions),
             "next_observation": np.asarray(following), "executed_ctrl": np.asarray(controls),
             "ik_error_m": np.asarray(ik_errors), "phase": np.asarray(phases)}
    before_cut = trace["observation"][:, 7] == 0
    target_distances = np.linalg.norm(trace["next_observation"][:, :3], axis=1) / 10
    cut_steps = np.flatnonzero(trace["next_observation"][:, 7] > 0)
    result.update(outcome=outcome(result, phase), recovery=recovery,
                  initialization_failed=False, reopened_before_cut=bool(reopened),
                  environment=env.configuration(), phases=dict(Counter(phases)),
                  first_cut_step=int(cut_steps[0] + 1) if len(cut_steps) else None,
                  final_target_distance_m=float(np.linalg.norm(env.target - env.tip)),
                  final_home_distance_m=float(np.linalg.norm(env.home - env.tip)),
                  min_precut_target_distance_m=float(target_distances[before_cut].min()),
                  final_opening_fraction=float(obs[6]), max_ik_error_m=float(max(ik_errors)),
                  last20_tool_displacement_m=float(np.linalg.norm(
                      trace["next_observation"][-1, 3:6] - trace["observation"][-min(20, len(phases)), 3:6])))
    return result, trace


def summarize(rows):
    completed = [r for r in rows if not r.get("initialization_failed", False)]
    successes = [r for r in completed if r["success"]]
    return {"count": len(rows), "success_count": len(successes),
            "success_rate": len(successes) / len(rows),
            "initialization_failures": len(rows) - len(completed),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "target_cut_count": sum(r["target_cut"] for r in completed),
            "wrong_cut_episodes": sum(r["wrong_cuts"] > 0 for r in completed),
            "protected_intrusion_episodes": sum(r["protected_intrusions"] > 0 for r in completed),
            "tool_terrain_contact_episodes": sum(r["tool_terrain_contact"] for r in completed),
            "time_limit_episodes": sum(r["steps"] >= 180 and not r["success"] for r in completed),
            "timeout_before_cut": sum(r["outcome"] == "timeout" and not r["target_cut"] for r in completed),
            "timeout_after_cut": sum(r["outcome"] == "timeout" and r["target_cut"] for r in completed),
            "intrusion_before_cut": sum(r["protected_intrusions"] > 0 and not r["target_cut"] for r in completed),
            "intrusion_after_cut": sum(r["protected_intrusions"] > 0 and r["target_cut"] for r in completed),
            "mean_sim_seconds": float(np.mean([r["sim_seconds"] for r in completed])) if completed else None,
            "success_mean_sim_seconds": float(np.mean([r["sim_seconds"] for r in successes])) if successes else None}


def evaluate(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {out}")
    policy = TreeBC(args.checkpoint)
    if policy.observation_schema != SCHEMA:
        raise ValueError("Stage 4 requires a state30_v2 checkpoint")
    report = {**metadata(), "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "seed_start": args.seed, "count_per_condition": args.episodes,
              "recovery": args.recovery, "ood": False, "max_steps": 180,
              "simulation_seconds_limit": 180 * DT, "conditions": [], "complete": False,
              "bc_action_source": "TreeBC.predict(observation); no teacher corrections",
              "independent_layouts": args.episodes,
              "layout_reuse": "same seeds across all nine shape/tilt conditions"}
    report["source_sha256"]["stage4.py"] = sha256((ROOT / "stage4.py").read_bytes()).hexdigest()
    started = time.monotonic()
    for plant, tilt in CONDITIONS:
        for label, controller in (("waypoint_v2", None), ("bc_state30_v1", policy)):
            env = CuttingEnv(plant=plant, tilt=tilt, geometry="stem_segment_v1",
                             teacher="waypoint_v2", observation_schema=SCHEMA, max_steps=180)
            rows = []
            try:
                for i in range(args.episodes):
                    env.azimuth = float(i % 4 * 90)
                    row, trace = rollout(env, args.seed + i, controller, args.recovery)
                    row["policy"] = label
                    if not row["success"] and trace:
                        path = out.parent / (out.stem + "_failures") / f"{label}_{plant}_{tilt}_{args.seed+i}.npz"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(path, **trace)
                        row["failure_trace"] = str(path)
                    rows.append(row)
                report["conditions"].append({"plant": plant, "tilt": tilt, "policy": label,
                                             **summarize(rows), "episodes": rows})
                print(plant, tilt, label, summarize(rows), flush=True)
            finally:
                env.close()
            report["elapsed_seconds"] = time.monotonic() - started
            save_json(out, report)
    report["summary"] = {label: summarize([r for c in report["conditions"] if c["policy"] == label
                                           for r in c["episodes"]])
                         for label in ("waypoint_v2", "bc_state30_v1")}
    report["complete"] = True
    save_json(out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
