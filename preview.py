"""Roadmap stages 1/2: frozen BC background comparison and plant smoke tests.

All rollouts, including failures, are recorded. Rendering does not modify state.
This is a development preview, not a new training run or held-out benchmark.
"""
import argparse
from hashlib import sha256
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from environment import CuttingEnv, ROOT
from run import TreeBC, provenance, save_json
from scenery import CAMERAS


def caption(frame, title, detail):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 48), fill=(21, 33, 33))
    draw.text((12, 6), title, font=ImageFont.load_default(size=17), fill=(245, 237, 212))
    draw.text((12, 29), detail, font=ImageFont.load_default(size=12), fill=(184, 217, 202))
    return image


def sheet(images, path):
    if not images:
        return
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 3, height * ((len(images) + 2) // 3)))
    for i, image in enumerate(images):
        canvas.paste(image, ((i % 3) * width, (i // 3) * height))
    canvas.save(path)


def failure_reason(result):
    if result["success"]:
        return "success"
    if result["wrong_cuts"]:
        return "wrong_cut"
    if result["protected_intrusions"]:
        return "protected_intrusion"
    if result["tool_terrain_contact"]:
        return "terrain_contact"
    return "timeout"


def rollout(env, policy, seed, capture=False, writer=None):
    obs = env.reset(seed)
    digest = sha256(obs.tobytes())
    frames = {}
    if capture:
        frames = {cam: env.render(640, 400, camera=cam) for cam in CAMERAS}
    ids = f"target {env.target_id + 1}: gold | other plants: blue"
    if writer is not None:
        overview = caption(frames["overview"], f"{env.scene.upper()} | frozen BC", "Establishing still (1 s) | " + ids)
        for _ in range(20):
            writer.append_data(np.asarray(overview))
    previous_frame = frames.get("closeup")
    cut_step = None
    done = False
    while not done:
        action = env.expert() if policy is None else policy.predict(obs)
        obs, done, result = env.step(action)
        digest.update(obs.tobytes())
        digest.update(np.asarray(action).tobytes())
        digest.update(env.data.qpos.tobytes())
        if capture:
            frame = env.render(640, 400, camera="closeup")
            if result["target_cut"] and "cut" not in frames:
                frames["aligned"] = previous_frame
                frames["cut"] = frame
                cut_step = env.steps
            if cut_step is not None and env.steps == cut_step + 3:
                frames["falling"] = frame
            previous_frame = frame
            if writer is not None:
                status = "CUT / retreat" if result["target_cut"] else "approach / align / close"
                labelled = caption(frame, f"{env.scene.upper()} | {status}",
                                   f"1x simulation time | {env.data.time:.2f} s | " + ids)
                writer.append_data(np.asarray(labelled))
    if capture:
        frames["after"] = env.render(640, 400, camera="closeup")
        if writer is not None:
            ending = caption(frames["after"], failure_reason(result).upper(),
                             "Final still (1 s) | same trial | visual-only background")
            for _ in range(20):
                writer.append_data(np.asarray(ending))
    return {**result, "outcome": failure_reason(result), "trajectory_sha256": digest.hexdigest()}, frames


def summarize(rows):
    return {"count": len(rows), "success_count": sum(r["success"] for r in rows),
            "outcomes": {key: sum(r["outcome"] == key for r in rows) for key in
                         ("success", "protected_intrusion", "wrong_cut", "terrain_contact", "timeout")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "backgrounds", "plants"), default="all")
    parser.add_argument("--checkpoint", default="outputs/bc_v2.npz")
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--episodes", type=int, default=1, help="Development seeds per condition")
    parser.add_argument("--azimuth", type=float, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/roadmap_preview"))
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {**provenance(), "purpose": "development preview; no retraining",
              "seed_start": args.seed, "episodes_per_condition": args.episodes,
              "preview_seed": args.seed, "conditions": []}
    report["source_sha256"]["preview.py"] = sha256((ROOT / "preview.py").read_bytes()).hexdigest()
    render = not args.no_render
    gallery, comparisons, plants_before, plants_aligned, plants_after = [], [], [], [], []
    if args.mode in ("all", "backgrounds"):
        policy = TreeBC(args.checkpoint)
        report["checkpoint_sha256"] = sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        reference = []
        for scene in ("baseline", "curb", "tree", "river"):
            env = CuttingEnv(scene=scene)
            writer = None
            rows = []
            try:
                if render and scene != "baseline":
                    writer = imageio.get_writer(str(args.out / f"{scene}_bc.mp4"), fps=20,
                                               codec="libx264", quality=8, macro_block_size=16)
                for i in range(args.episodes):
                    capture = render and scene != "baseline" and i == 0
                    row, frames = rollout(env, policy, args.seed + i, capture, writer if i == 0 else None)
                    rows.append(row)
                    if scene == "baseline":
                        reference.append(row["trajectory_sha256"])
                    elif row["trajectory_sha256"] != reference[i]:
                        raise AssertionError(f"Background changed trajectory: {scene}, seed={args.seed + i}")
                    if capture:
                        for cam in CAMERAS:
                            gallery.append(caption(frames[cam], f"{scene.upper()} | {cam}",
                                                   f"seed {args.seed} | identical initial physical state"))
                        for phase in ("aligned", "falling", "after"):
                            frame = frames.get(phase, frames["after"])
                            label = phase if phase in frames else "no cut / " + row["outcome"]
                            comparisons.append(caption(frame, f"{scene.upper()} | {label}",
                                                       f"frozen BC | seed {args.seed} | {row['outcome']}"))
                report["conditions"].append({"policy": "bc", "environment": env.configuration(),
                                             **summarize(rows), "episodes": rows})
                print(f"background {scene}: {summarize(rows)}", flush=True)
            finally:
                env.close()
                if writer is not None:
                    writer.close()
        report["background_trajectories_identical"] = True
    if args.mode in ("all", "plants"):
        for plant in ("slender", "broad", "tall"):
            for tilt in (0, 10, 20):
                env = CuttingEnv(scene="tree", plant=plant, tilt=tilt,
                                 azimuth=args.azimuth, geometry="stem_segment_v1")
                rows = []
                try:
                    for i in range(args.episodes):
                        row, frames = rollout(env, None, args.seed + i, render and i == 0)
                        rows.append(row)
                        if render and i == 0:
                            title = f"{plant.upper()} | tilt {tilt} deg"
                            plants_before.append(caption(frames["closeup"], title, f"before | seed {args.seed} | expert, not BC"))
                            plants_aligned.append(caption(frames.get("aligned", frames["after"]), title,
                                "before closure" if "aligned" in frames else "no cut / " + row["outcome"]))
                            plants_after.append(caption(frames["after"], title, f"final | {row['outcome']} | same camera"))
                    report["conditions"].append({"policy": "expert", "environment": env.configuration(),
                                                 **summarize(rows), "episodes": rows})
                    print(f"plant {plant}, {tilt} deg: {summarize(rows)}", flush=True)
                finally:
                    env.close()
    for images, filename in ((gallery, "scenes_cameras.png"), (comparisons, "scenes_cut.png"),
                              (plants_before, "plants_before.png"), (plants_aligned, "plants_aligned.png"),
                              (plants_after, "plants_after.png")):
        sheet(images, args.out / filename)
    report["elapsed_seconds"] = time.monotonic() - started
    save_json(args.out / "manifest.json", report)
    print(f"Saved {args.out / 'manifest.json'} ({report['elapsed_seconds']:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
