"""Interactive viewer. macOS: use mjpython play.py, not python play.py."""
import argparse
from collections import deque
import time

import mujoco.viewer
import numpy as np

from environment import CuttingEnv, DT
from run import TreeBC


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["manual", "expert", "bc"], default="bc")
    p.add_argument("--checkpoint", default="outputs/bc_v2.npz")
    p.add_argument("--seed", type=int, default=20000)
    args = p.parse_args()
    env = CuttingEnv(max_steps=10000 if args.policy == "manual" else 180)
    policy = TreeBC(args.checkpoint) if args.policy == "bc" else None
    commands = deque()
    obs = env.reset(args.seed)
    action = np.array([0., 0., 0., 1.])
    moving_for = 0
    paused = False
    print("W/S: +X/-X  A/D: +Y/-Y  E/Q: +Z/-Z  C: close  O: open  R: next seed  SPACE: pause")
    print("Manual control is for debugging; use evaluate for quantitative comparison.")
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=commands.append) as viewer:
        viewer.cam.lookat[:] = [0.38, 0, 0.25]
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -25
        while viewer.is_running():
            started = time.monotonic()
            while commands:
                key = commands.popleft()
                if key == 32:
                    paused = not paused
                elif key == ord("R"):
                    args.seed += 1
                    obs = env.reset(args.seed)
                    paused = False
                    action[:] = [0, 0, 0, 1]
                    moving_for = 0
                elif key in [ord("C"), ord("O")]:
                    action[3] = 0 if key == ord("C") else 1
                elif key in map(ord, "WSADEQ"):
                    direction = {"W": [1, 0, 0], "S": [-1, 0, 0], "A": [0, 1, 0],
                                 "D": [0, -1, 0], "E": [0, 0, 1], "Q": [0, 0, -1]}
                    action[:3] = direction[chr(key)]
                    moving_for = 4
            if not paused:
                if args.policy == "manual":
                    if moving_for <= 0:
                        action[:3] = 0
                    moving_for -= 1
                    chosen = action
                else:
                    chosen = env.expert() if policy is None else policy.predict(obs)
                obs, done, result = env.step(chosen)
                if done:
                    print(result, flush=True)
                    paused = True
            viewer.sync()
            time.sleep(max(0, DT - (time.monotonic() - started)))
    env.close()


if __name__ == "__main__":
    main()
