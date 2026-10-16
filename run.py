"""Reproducible collect → train → evaluate → video command line workflow."""
import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import ExtraTreesRegressor

from environment import ACTION_NAMES, OBS_NAMES, CuttingEnv, DT, ROOT


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def provenance():
    files = ["environment.py", "run.py"]
    return {"python": platform.python_version(), "system": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ["mujoco", "numpy", "scikit-learn", "imageio"]},
            "source_sha256": {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
            "model_commit": "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35",
            "control_hz": 1 / DT, "observations": OBS_NAMES, "actions": ACTION_NAMES}


def collect(args):
    env = CuttingEnv()
    obs, acts, next_obs, ids, seeds, rows = [], [], [], [], [], []
    for episode in range(args.episodes):
        seed = args.seed + episode
        current = env.reset(seed)
        episode_data = []
        done = False
        while not done:
            action = env.expert()
            following, done, result = env.step(action)
            episode_data.append((current, action, following))
            current = following
        rows.append(result)
        # Failed trajectories remain in the manifest but are not positive BC demos.
        if result["success"]:
            for o, a, n in episode_data:
                obs.append(o); acts.append(a); next_obs.append(n)
                ids.append(episode); seeds.append(seed)
        if episode % 20 == 0 or episode == args.episodes - 1:
            print(f"collect {episode+1}/{args.episodes} successful={sum(r['success'] for r in rows)} frames={len(obs)}", flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not obs:
        raise RuntimeError("Expert produced zero successful trajectories. Inspect environment before training.")
    np.savez_compressed(path, observation=np.asarray(obs), action=np.asarray(acts),
                        next_observation=np.asarray(next_obs), episode_id=np.asarray(ids), seed=np.asarray(seeds))
    save_json(path.with_suffix(".json"), {**provenance(), "requested_episodes": args.episodes,
              "success_count": sum(r["success"] for r in rows), "frames": len(obs),
              "dataset_sha256": sha256(path.read_bytes()).hexdigest(), "episodes": rows})
    env.close()


class TreeBC:
    """Pure numeric checkpoint: no pickle/untrusted Python deserialization.

    State-based behavior cloning with an ensemble of decision-tree regressors.
    Each action comes solely from the learned trees and current observation.
    This is supervised imitation from scratch, not a pretrained VLA.
    """
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.left = z["left"]; self.right = z["right"]
            self.feature = z["feature"]; self.threshold = z["threshold"]
            self.values = z["values"]; self.starts = z["starts"]

    def predict(self, obs):
        nodes = self.starts.copy()
        active = self.left[nodes] >= 0
        while np.any(active):
            ids = nodes[active]
            nodes[active] = np.where(obs[self.feature[ids]] <= self.threshold[ids],
                                    self.left[ids], self.right[ids])
            active = self.left[nodes] >= 0
        return self.values[nodes].mean(axis=0)


def train(args):
    with np.load(args.data, allow_pickle=False) as z:
        x, y, episodes = z["observation"], z["action"], z["episode_id"]
    unique = np.unique(episodes)
    if len(unique) < 5:
        raise ValueError("Need at least 5 successful episodes for episode-disjoint validation.")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    val_ids = unique[:max(1, len(unique) // 5)]
    val = np.isin(episodes, val_ids)
    # ExtraTrees is a small CPU-only BC baseline, selected for fast Mac training.
    model = ExtraTreesRegressor(n_estimators=48, max_depth=22, min_samples_leaf=2,
                               random_state=args.seed, n_jobs=2)
    started = time.monotonic()
    model.fit(x[~val], y[~val])
    mse = np.mean((model.predict(x[val]) - y[val]) ** 2, axis=0)
    left, right, feature, threshold, values, starts = [], [], [], [], [], []
    offset = 0
    for est in model.estimators_:
        t = est.tree_
        starts.append(offset)
        left.append(np.where(t.children_left < 0, -1, t.children_left + offset))
        right.append(np.where(t.children_right < 0, -1, t.children_right + offset))
        feature.append(t.feature); threshold.append(t.threshold); values.append(t.value[:, :, 0])
        offset += t.node_count
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, left=np.concatenate(left), right=np.concatenate(right),
                        feature=np.concatenate(feature), threshold=np.concatenate(threshold),
                        values=np.concatenate(values), starts=np.asarray(starts))
    # Verify numeric checkpoint inference agrees with fitted sklearn model.
    loaded = TreeBC(out)
    check_x = x[val][::max(1, int(val.sum()) // 20)]
    np.testing.assert_allclose(np.asarray([loaded.predict(o) for o in check_x]), model.predict(check_x), atol=1e-7)
    info = {**provenance(), "algorithm": "ExtraTrees behavior cloning (from scratch)",
            "train_episode_ids": unique[len(val_ids):].tolist(), "validation_episode_ids": val_ids.tolist(),
            "train_frames": int((~val).sum()), "validation_frames": int(val.sum()),
            "validation_action_mse": mse.tolist(), "seed": args.seed,
            "elapsed_seconds": time.monotonic() - started,
            "data_sha256": sha256(Path(args.data).read_bytes()).hexdigest(),
            "checkpoint_sha256": sha256(out.read_bytes()).hexdigest()}
    save_json(out.with_suffix(".json"), info)
    print(json.dumps({k: info[k] for k in ["algorithm", "train_frames", "validation_frames", "validation_action_mse", "elapsed_seconds"]}, indent=2))


def evaluate(args):
    env = CuttingEnv()
    policy = TreeBC(args.checkpoint) if args.policy == "bc" else None
    rows = []
    for i in range(args.episodes):
        obs = env.reset(args.seed + i, args.ood)
        done = False
        trajectory = []
        while not done:
            action = policy.predict(obs) if policy is not None else env.expert()
            previous = obs
            obs, done, result = env.step(action)
            trajectory.append({"observation": previous.tolist(), "policy_action": np.asarray(action).tolist(),
                               "executed_ctrl": env.data.ctrl.tolist(), "ik_error_m": env.last_ik_error})
        rows.append(result)
        if args.logs:
            save_json(Path(args.logs) / f"seed_{args.seed+i}.json", {"result": result, "trajectory": trajectory})
        print(f"{args.policy} seed={args.seed+i} success={result['success']} cut={result['target_cut']} steps={result['steps']}", flush=True)
    count = len(rows)
    report = {**provenance(), "policy": args.policy, "ood": args.ood, "episodes": rows,
              "count": count, "success_count": sum(r["success"] for r in rows),
              "success_rate": sum(r["success"] for r in rows) / count,
              "cut_rate": sum(r["target_cut"] for r in rows) / count,
              "wrong_cut_episodes": sum(r["wrong_cuts"] > 0 for r in rows),
              "protected_intrusion_episodes": sum(r["protected_intrusions"] > 0 for r in rows),
              "tool_terrain_contact_episodes": sum(r["tool_terrain_contact"] for r in rows),
              "mean_sim_seconds": float(np.mean([r["sim_seconds"] for r in rows]))}
    if args.policy == "bc":
        report["checkpoint_sha256"] = sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    save_json(args.out, report)
    env.close()
    print(f"Saved {args.out}: {report['success_count']}/{count}", flush=True)


def overlay(frame, title, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    draw.rectangle((0, 0, img.width, 86), fill=(16, 29, 32))
    draw.text((22, 12), title, font=font, fill=(246, 242, 223))
    for j, line in enumerate(lines):
        draw.text((22, 40 + j * 20), line, font=small, fill=(162, 213, 203))
    return np.asarray(img)


def video(args):
    env = CuttingEnv()
    policy = TreeBC(args.checkpoint) if args.policy == "bc" else None
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with imageio.get_writer(str(path), fps=20, codec="libx264", quality=8, macro_block_size=16) as writer:
        for i in range(args.episodes):
            obs = env.reset(args.seed + i, args.ood)
            for closeup in [False, True]:
                if closeup:
                    obs = env.reset(args.seed + i, args.ood)
                done = False
                while not done:
                    action = policy.predict(obs) if policy is not None else env.expert()
                    obs, done, result = env.step(action)
                    state = "CUT - returning" if result["target_cut"] else "Approach / align / close"
                    frame = overlay(env.render(closeup=closeup), "SELECTIVE WEED CUTTING | " + args.policy.upper(),
                        [f"Seed {args.seed+i} | target {env.target_id+1} (gold marker) | {env.data.time:.2f} s | {state}",
                         "State-based policy | geometric cut event | 1x simulation time | close-up replay" if closeup
                         else "Panda + actuated scissors | gold = target, blue = protected | 1x simulation time"])
                    writer.append_data(frame)
                    if i == 0 and not closeup and env.steps == 1:
                        Image.fromarray(frame).save(path.with_suffix(".png"))
                    if result["target_cut"] and not path.with_name(path.stem + "_cut.png").exists():
                        Image.fromarray(frame).save(path.with_name(path.stem + "_cut.png"))
                for _ in range(20):
                    writer.append_data(overlay(env.render(closeup=closeup),
                        "SUCCESS" if result["success"] else "FAILED / TIMEOUT",
                        [f"target cut: {result['target_cut']} | wrong cuts: {result['wrong_cuts']} | protected intrusions: {result['protected_intrusions']}",
                         "Simulated cutting skill. Plant fracture forces and real-world transfer are not validated."]))
            results.append(result)
            print(f"video seed={args.seed+i}: {result['success']}", flush=True)
    save_json(path.with_suffix(".json"), {**provenance(), "policy": args.policy, "results": results})
    env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--episodes", type=int, default=200)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default="outputs/demos.npz")
    t = sub.add_parser("train")
    t.add_argument("--data", default="outputs/demos.npz")
    t.add_argument("--out", default="outputs/bc.npz")
    t.add_argument("--seed", type=int, default=42)
    for name in ["evaluate", "video"]:
        p = sub.add_parser(name)
        p.add_argument("--policy", choices=["expert", "bc"], default="expert")
        p.add_argument("--checkpoint", default="outputs/bc.npz")
        p.add_argument("--episodes", type=int, default=30 if name == "evaluate" else 3)
        p.add_argument("--seed", type=int, default=10000)
        p.add_argument("--ood", action="store_true")
        p.add_argument("--out", default="outputs/evaluation.json" if name == "evaluate" else "outputs/demo.mp4")
        if name == "evaluate":
            p.add_argument("--logs", default="")
    args = parser.parse_args()
    if getattr(args, "episodes", 1) <= 0:
        parser.error("--episodes must be positive")
    {"collect": collect, "train": train, "evaluate": evaluate, "video": video}[args.command](args)


if __name__ == "__main__":
    main()
