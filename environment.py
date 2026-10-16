"""MuJoCo Panda + one driven scissor hinge + procedural stems.

Cutting is a geometric event, NOT fracture/force simulation. Plant geoms do
not collide with the blades; protected-plant intersections are measured
geometrically at every physics substep. The arm is advanced with mj_step.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
PANDA = ROOT / "vendor" / "franka_emika_panda"
DT = 0.05
SUBSTEPS = 25
MAX_DELTA = 0.008
OPEN_ANGLE = 0.65
CUT_ANGLE = 0.12
CUT_TOLERANCE = 0.012
OBS_NAMES = ([f"target_minus_tool_{a}_dm" for a in "xyz"]
             + [f"tool_world_{a}_m" for a in "xyz"]
             + ["opening_fraction", "target_cut"]
             + [f"joint{i}_rad" for i in range(1, 8)])
ACTION_NAMES = ["dx_over_8mm", "dy_over_8mm", "dz_over_8mm", "opening_fraction"]


def vec(value):
    return " ".join(f"{x:.8g}" for x in value)


def el(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def cutting_intersection(local, before, after, radius=0.003):
    """Stem center in pivot coordinates, closure edge, finite blade region."""
    x, y, z = local
    return bool(before > CUT_ANGLE >= after
                and 0.035 <= x <= 0.135
                and abs(y - 0.004) <= 0.006 + radius
                and abs(z) <= CUT_TOLERANCE)


def scene_xml():
    source = PANDA / "panda_nohand.xml"
    if not source.exists():
        raise FileNotFoundError("Run python fetch_assets.py before running the simulation.")
    root = ET.parse(source).getroot()
    root.set("model", "Selective weed cutting — Panda")
    root.find("compiler").set("meshdir", str(PANDA / "assets"))
    root.find("option").set("timestep", "0.002")
    # Reset is handled explicitly: the added scissor/free joints change nq.
    root.remove(root.find("keyframe"))
    visual = el(root, "visual")
    el(visual, "global", offwidth="1280", offheight="720")
    el(visual, "quality", shadowsize="2048")
    el(visual, "headlight", diffuse="0.5 0.5 0.5", ambient="0.3 0.3 0.3")
    asset = root.find("asset")
    el(asset, "texture", name="ground_tex", type="2d", builtin="checker",
       rgb1="0.22 0.28 0.26", rgb2="0.25 0.31 0.29", width="512", height="512")
    el(asset, "material", name="ground_mat", texture="ground_tex", texrepeat="8 8", reflectance="0.1")
    world = root.find("worldbody")
    el(world, "geom", name="floor", type="plane", size="2 2 0.02", material="ground_mat", conaffinity="5")
    el(world, "light", pos="0.3 -1 2", dir="0 0.5 -1", diffuse="0.8 0.8 0.8")
    el(world, "geom", name="soil", type="box", pos="0.57 0 0.015",
       size="0.17 0.27 0.015", rgba="0.24 0.16 0.09 1", conaffinity="5")
    for y in [-0.29, 0.29]:
        el(world, "geom", name=f"curb_{y}", type="box", pos=f"0.57 {y} 0.024",
           size="0.19 0.02 0.024", rgba="0.56 0.59 0.57 1")

    attachment = root.find(".//body[@name='attachment']")
    tool = el(attachment, "body", name="scissors", pos="0 0 0.035", quat="0 1 0 0")
    el(tool, "geom", name="tool_mount", type="cylinder", size="0.025 0.035",
       pos="0 0 0.035", rgba="0.14 0.18 0.20 1", mass="0.08")
    el(tool, "geom", name="pivot_visual", type="cylinder", size="0.012 0.012",
       rgba="0.05 0.65 0.70 1", mass="0.02")
    el(tool, "geom", name="fixed_blade", type="box", pos="0.075 0 -0.002",
       size="0.075 0.004 0.002", rgba="0.75 0.82 0.87 1", mass="0.03")
    el(tool, "site", name="cut_site", pos="0.085 0.004 0", size="0.003",
       rgba="1 0.75 0.15 0.6")
    moving = el(tool, "body", name="moving_blade")
    el(moving, "joint", name="scissor_joint", type="hinge", axis="0 0 1",
       range=f"0 {OPEN_ANGLE}", damping="0.05", armature="0.002")
    el(moving, "geom", name="moving_blade_geom", type="box", pos="0.075 0 0.003",
       size="0.075 0.004 0.002", rgba="0.9 0.92 0.95 1", mass="0.03")
    for parent in [tool, moving]:
        el(parent, "geom", type="capsule", fromto="-0.06 0 0 0 0 0",
           size="0.007", rgba="0.04 0.50 0.55 1", contype="0", conaffinity="0", mass="0.01")
    el(root.find("actuator"), "position", name="scissor_actuator", joint="scissor_joint",
       kp="8", kv="0.2", ctrlrange=f"0 {OPEN_ANGLE}", forcerange="-2 2")
    contact = root.find("contact")
    el(contact, "exclude", body1="scissors", body2="moving_blade")
    el(contact, "exclude", body1="attachment", body2="scissors")
    equality = el(root, "equality")
    for i in range(3):
        x, y, z = 0.56, (i - 1) * 0.10, 0.14
        anchor = el(world, "body", name=f"root{i}", pos=f"{x} {y} 0.03")
        el(anchor, "geom", name=f"stump{i}", type="capsule", fromto="0 0 0 0 0 0.11",
           size="0.003", rgba="0.27 0.53 0.16 1", contype="0", conaffinity="0")
        el(anchor, "geom", name=f"marker{i}", type="cylinder", size="0.021 0.002",
           pos="0 0 0.003", rgba="0.17 0.38 0.19 1", contype="0", conaffinity="0")
        upper = el(world, "body", name=f"upper{i}", pos=f"{x} {y} {z}")
        el(upper, "freejoint", name=f"upper_joint{i}")
        el(upper, "geom", name=f"stem{i}", type="capsule", fromto="0 0 0 0 0 0.14",
           size="0.003", mass="0.008", rgba="0.28 0.68 0.23 1", contype="0", conaffinity="0")
        for j, sign in enumerate([-1, 1]):
            el(upper, "geom", name=f"leaf{i}_{j}", type="ellipsoid",
               pos=f"{sign * 0.018} 0 {0.07 + j * 0.04}", size="0.028 0.007 0.009",
               euler=f"0 {sign * -0.45} 0", mass="0.001", rgba="0.25 0.60 0.20 1",
               contype="0", conaffinity="0")
        el(equality, "weld", name=f"stem_weld{i}", body1=f"root{i}", body2=f"upper{i}", solref="0.002 1")
    return ET.tostring(root, encoding="unicode")


class CuttingEnv:
    def __init__(self, max_steps=180):
        self.model = mujoco.MjModel.from_xml_string(scene_xml())
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.site_id = self.model.site("cut_site").id
        self.tool_id = self.model.body("scissors").id
        self.scissor_q = self.model.jnt_qposadr[self.model.joint("scissor_joint").id]
        self.arm_q = np.array([self.model.jnt_qposadr[self.model.joint(f"joint{i}").id] for i in range(1, 8)])
        self.arm_v = np.array([self.model.jnt_dofadr[self.model.joint(f"joint{i}").id] for i in range(1, 8)])
        self.limits = np.array([self.model.joint(f"joint{i}").range for i in range(1, 8)])
        self.upper_q = [self.model.jnt_qposadr[self.model.joint(f"upper_joint{i}").id] for i in range(3)]
        self.eq_ids = [self.model.equality(f"stem_weld{i}").id for i in range(3)]
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros_like(self.jacp)
        self.renderer = None
        self.rest_q = np.array([0, -0.4, 0, -2.1, 0, 1.8, -0.785])

    def solve_ik(self, pos, start, iterations=18):
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        q = start.copy()
        residual = 0.0
        for _ in range(iterations):
            d.qpos[self.arm_q] = q
            mujoco.mj_forward(self.model, d)
            error_p = pos - d.site_xpos[self.site_id]
            # Desired tool orientation is identity: blades lie horizontally.
            rot = d.site_xmat[self.site_id].reshape(3, 3)
            error_r = Rotation.from_matrix(rot.T).as_rotvec()
            residual = np.linalg.norm(error_p)
            if residual < 0.0003 and np.linalg.norm(error_r) < 0.003:
                break
            mujoco.mj_jacSite(self.model, d, self.jacp, self.jacr, self.site_id)
            jac = np.vstack([self.jacp[:, self.arm_v], self.jacr[:, self.arm_v] * 0.3])
            error = np.r_[error_p, error_r * 0.3]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.0002 * np.eye(6), error)
            q = np.clip(q + np.clip(dq, -0.15, 0.15), self.limits[:, 0] + 0.01, self.limits[:, 1] - 0.01)
        return q, float(residual)

    def reset(self, seed=0, ood=False):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.seed, self.ood = seed, ood
        self.steps = 0
        self.cut = np.zeros(3, dtype=bool)
        self.protected_touched = set()
        self.terrain_contact = False
        self.action_clips = 0
        self.events = []
        self.target_id = int(rng.integers(3))
        self.stems = np.zeros((3, 3))
        for i in range(3):
            x = rng.uniform(0.52, 0.58) + (0.035 if ood else 0)
            y = (i - 1) * 0.10 + rng.uniform(-0.015, 0.015)
            height = rng.uniform(0.13, 0.17) + (0.025 if ood else 0)
            self.stems[i] = [x, y, height]
            body = self.model.body(f"root{i}")
            body.pos[:] = [x, y, 0.03]
            geom = self.model.geom(f"stump{i}")
            geom.pos[:] = [0, 0, (height - 0.03) / 2]
            geom.size[1] = (height - 0.03) / 2
            q = self.upper_q[i]
            self.data.qpos[q:q + 7] = [x, y, height, 1, 0, 0, 0]
            self.model.body(f"upper{i}").pos[:] = [x, y, height]
            eq = self.eq_ids[i]
            self.model.eq_data[eq, 3:10] = [0, 0, height - 0.03, 1, 0, 0, 0]
            self.data.eq_active[eq] = True
            self.model.geom(f"stem{i}").contype[:] = 0
            self.model.geom(f"stem{i}").conaffinity[:] = 0
            target = i == self.target_id
            self.model.geom(f"marker{i}").rgba[:] = [0.95, 0.65, 0.1, 1] if target else [0.2, 0.65, 0.80, 1]
            self.model.geom(f"stem{i}").rgba[:] = [0.55, 0.75, 0.18, 1] if target else [0.14, 0.55, 0.34, 1]
        self.target = self.stems[self.target_id].copy()
        self.home = np.array([0.38, -0.01, 0.31])
        self.data.qpos[self.scissor_q] = OPEN_ANGLE
        self.data.qpos[self.arm_q] = self.rest_q
        q, residual = self.solve_ik(self.home, self.rest_q, iterations=150)
        if residual > 0.003:
            raise RuntimeError(f"Home IK failed: {residual:.4f} m")
        self.data.qpos[self.arm_q] = q
        self.data.ctrl[:7] = q
        self.data.ctrl[7] = OPEN_ANGLE
        mujoco.mj_forward(self.model, self.data)
        self.previous_angle = float(self.data.qpos[self.scissor_q])
        self.last_ik_error = 0.0
        return self.observe()

    @property
    def tip(self):
        return self.data.site_xpos[self.site_id].copy()

    def observe(self):
        return np.r_[(self.target - self.tip) * 10, self.tip,
                     self.data.qpos[self.scissor_q] / OPEN_ANGLE,
                     float(self.cut[self.target_id]), self.data.qpos[self.arm_q]].astype(np.float32)

    def expert(self):
        # Expert alone sees the selected target; evaluation BC calls predict(observation).
        if self.cut[self.target_id]:
            goal = self.home
            opening = 1.0 if self.tip[2] > self.target[2] + 0.07 else 0.0
        else:
            goal = self.target
            opening = 0.0 if np.linalg.norm(self.tip - goal) < 0.003 else 1.0
        delta = np.clip((goal - self.tip) * 0.45 / MAX_DELTA, -1, 1)
        return np.r_[delta, opening].astype(np.float32)

    def _events(self):
        angle = float(self.data.qpos[self.scissor_q])
        pos = self.data.xpos[self.tool_id]
        rot = self.data.xmat[self.tool_id].reshape(3, 3)
        for i, stem in enumerate(self.stems):
            local = rot.T @ (stem - pos)
            if not self.cut[i] and cutting_intersection(local, self.previous_angle, angle):
                self.cut[i] = True
                self.data.eq_active[self.eq_ids[i]] = False
                # Only floor/soil accept bit 4: detached stems can land without
                # suddenly adding an unmodelled blade/plant cutting-force contact.
                self.model.geom(f"stem{i}").contype[:] = 4
                self.events.append({"time_s": float(self.data.time), "plant": i,
                                    "target": i == self.target_id, "height_error_m": abs(float(local[2]))})
            # Conservative cutting-envelope intrusion, not a measured force.
            if i != self.target_id and 0.02 < local[0] < 0.155 and abs(local[2]) < 0.018:
                if -0.01 < local[1] < local[0] * np.sin(max(angle, 0)) + 0.01:
                    self.protected_touched.add(i)
        for c in self.data.contact:
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in [c.geom1, c.geom2]]
            # Base/floor contact is excluded: only the cutting tool is checked here.
            if any(n in ["fixed_blade", "moving_blade_geom", "tool_mount"] for n in names):
                if any(n == "floor" or n == "soil" or n.startswith("curb_") for n in names):
                    self.terrain_contact = True
        self.previous_angle = angle

    def step(self, action):
        raw = np.asarray(action, dtype=float)
        if raw.shape != (4,) or not np.isfinite(raw).all():
            raise ValueError("Action must contain four finite numbers.")
        action = np.clip(raw, [-1, -1, -1, 0], [1, 1, 1, 1])
        self.action_clips += int(not np.array_equal(raw, action))
        desired = np.clip(self.tip + action[:3] * MAX_DELTA, [0.28, -0.24, 0.095], [0.72, 0.24, 0.43])
        q, self.last_ik_error = self.solve_ik(desired, self.data.qpos[self.arm_q])
        # Independent joint-setpoint rate bound; plant interaction remains geometric.
        self.data.ctrl[:7] = self.data.qpos[self.arm_q] + np.clip(q - self.data.qpos[self.arm_q], -0.08, 0.08)
        self.data.ctrl[7] = action[3] * OPEN_ANGLE
        for _ in range(SUBSTEPS):
            # Simulation-only gravity compensation; arm positions are not teleported.
            self.data.qfrc_applied[self.arm_v] = self.data.qfrc_bias[self.arm_v]
            mujoco.mj_step(self.model, self.data)
            self._events()
        self.steps += 1
        returned = np.linalg.norm(self.tip - self.home) < 0.012
        done = bool(self.steps >= self.max_steps or (self.cut[self.target_id] and returned)
                    or self.cut[np.arange(3) != self.target_id].any() or self.protected_touched or self.terrain_contact)
        return self.observe(), done, self.result()

    def result(self):
        wrong = int(self.cut[np.arange(3) != self.target_id].sum())
        returned = bool(np.linalg.norm(self.tip - self.home) < 0.012)
        success = bool(self.cut[self.target_id] and not wrong and not self.protected_touched
                       and not self.terrain_contact and returned)
        return {"seed": int(self.seed), "ood": bool(self.ood), "target_id": self.target_id,
                "success": success, "target_cut": bool(self.cut[self.target_id]),
                "wrong_cuts": wrong, "protected_intrusions": len(self.protected_touched),
                "tool_terrain_contact": self.terrain_contact, "returned": returned,
                "steps": self.steps, "sim_seconds": round(self.steps * DT, 3),
                "action_clips": self.action_clips, "events": self.events,
                "target_xyz": self.target.tolist()}

    def render(self, width=960, height=640, closeup=False):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.48, 0, 0.23] if closeup else [0.32, 0, 0.35]
        cam.distance = 0.75 if closeup else 1.45
        cam.azimuth = 130
        cam.elevation = -26
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
