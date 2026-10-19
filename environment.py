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

from scenery import SCENES, PLANTS, camera_settings, decorate

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
GEOMETRIES = ("legacy_point_v1", "stem_segment_v1")
OBS_V2_NAMES = OBS_NAMES + [f"target_direction_{a}" for a in "xyz"] + [
    f"protected{i}_{endpoint}_minus_tool_{a}_dm"
    for i in range(2) for endpoint in ("root", "top") for a in "xyz"]
OBS_SCHEMAS = {"state15_v1": OBS_NAMES, "state30_v2": OBS_V2_NAMES}


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


def segment_plane_intersection(start, end):
    """Intersection with tool-local z=0; parallel/non-crossing stems miss."""
    delta = end - start
    if abs(delta[2]) < 1e-10:
        return None
    fraction = -start[2] / delta[2]
    return start + fraction * delta if 0 <= fraction <= 1 else None


def segment_cut_intersection(start, end, joint, before, after, radius=0.003):
    """Finite stem/blade intersection, restricted to the pre-modelled seam.

    We still detach at the fixed weld, not at an arbitrary new mesh location.
    The plane hit must be within 12 mm along the stem of that seam.
    """
    hit = segment_plane_intersection(start, end)
    if hit is None or np.linalg.norm(hit - joint) > CUT_TOLERANCE:
        return None
    return hit if cutting_intersection(hit, before, after, radius) else None


def segment_protected_intersection(start, end, angle, radius=0.003):
    """Clip a full stem segment against a conservative open-blade envelope."""
    slope = np.sin(max(angle, 0))
    normals = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1],
                        [0, 0, 1], [0, -1, 0], [-slope, 1, 0]])
    bounds = np.array([-0.02 + radius, 0.155 + radius, 0.018 + radius,
                       0.018 + radius, 0.01 + radius, 0.01 + radius])
    low, high = 0.0, 1.0
    for normal, bound in zip(normals, bounds):
        offset = bound - normal @ start
        rate = normal @ (end - start)
        if abs(rate) < 1e-12:
            if offset < 0:
                return False
        elif rate > 0:
            high = min(high, offset / rate)
        else:
            low = max(low, offset / rate)
        if low > high:
            return False
    return True


def scene_xml(scene="baseline", plant="legacy"):
    if scene not in SCENES or plant not in PLANTS:
        raise ValueError("Unknown scene or plant type")
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
    decorate(root, scene)

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
        # World marker stays horizontal even when the root body tilts.
        el(world, "geom", name=f"marker{i}", type="cylinder", size="0.021 0.002",
           pos=f"{x} {y} 0.033", rgba="0.17 0.38 0.19 1", contype="0", conaffinity="0")
        upper = el(world, "body", name=f"upper{i}", pos=f"{x} {y} {z}")
        el(upper, "freejoint", name=f"upper_joint{i}")
        length = 0.20 if plant == "tall" else 0.14
        el(upper, "geom", name=f"stem{i}", type="capsule", fromto=f"0 0 0 0 0 {length}",
           # Compile terrain-only contact into body masks and collision BVHs.
           # The uncut upper is welded above ground; tools never accept bit 4.
           size="0.003", mass="0.008", rgba="0.28 0.68 0.23 1", contype="4", conaffinity="0",
           priority="1", solref="0.004 1", solimp="0.99 0.999 0.001", margin="0.001")
        for j, sign in enumerate([-1, 1]):
            leaf_size = {"legacy": "0.028 0.007 0.009", "slender": "0.045 0.003 0.004",
                         "broad": "0.036 0.018 0.006", "tall": "0.021 0.005 0.004"}[plant]
            el(upper, "geom", name=f"leaf{i}_{j}", type="ellipsoid",
               pos=f"{sign * 0.018} 0 {0.07 + j * 0.04}", size=leaf_size,
               euler=f"0 {sign * -0.45} 0", mass="0.001", rgba="0.25 0.60 0.20 1",
               contype="0", conaffinity="0")
        if plant != "legacy":
            # Visual-only secondary leaves; same rigid stem collision primitive.
            for j in range(3):
                azimuth = j * 2.4 + i * 0.4
                r = 0.022 if plant == "broad" else 0.014
                el(upper, "geom", type="ellipsoid", name=f"foliage{i}_{j}",
                   pos=vec([r * np.cos(azimuth), r * np.sin(azimuth), 0.035 + j * 0.038]),
                   size=leaf_size, euler=vec([0, -0.5, azimuth]), mass="0",
                   rgba="0.30 0.51 0.15 1", contype="0", conaffinity="0")
            if plant == "tall":
                el(upper, "geom", type="ellipsoid", name=f"seedhead{i}",
                   pos="0 0 0.19", size="0.009 0.009 0.029", mass="0",
                   rgba="0.57 0.48 0.22 1", contype="0", conaffinity="0")
        el(equality, "weld", name=f"stem_weld{i}", body1=f"root{i}", body2=f"upper{i}", solref="0.002 1")
    return ET.tostring(root, encoding="unicode")


class CuttingEnv:
    def __init__(self, max_steps=180, scene="baseline", plant="legacy", tilt=0,
                 azimuth=0, geometry="legacy_point_v1", teacher="direct_v1",
                 observation_schema="state15_v1"):
        if teacher not in ("direct_v1", "waypoint_v2") or observation_schema not in OBS_SCHEMAS:
            raise ValueError("Unknown teacher or observation schema")
        if teacher == "waypoint_v2" and geometry != "stem_segment_v1":
            raise ValueError("waypoint_v2 requires stem_segment_v1")
        self.teacher, self.observation_schema = teacher, observation_schema
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}")
        if not np.isfinite(tilt) or not 0 <= tilt <= 20 or not np.isfinite(azimuth):
            raise ValueError("tilt must be 0..20 degrees and azimuth must be finite")
        if tilt != 0 and geometry != "stem_segment_v1":
            raise ValueError("Tilt requires geometry='stem_segment_v1'")
        self.scene, self.plant = scene, plant
        self.tilt, self.azimuth, self.geometry = float(tilt), float(azimuth), geometry
        self.upper_length = 0.20 if plant == "tall" else 0.14
        self.model = mujoco.MjModel.from_xml_string(scene_xml(scene, plant))
        if geometry == "stem_segment_v1":
            # Tilted stems load the weld in rotation: hold the seam firmly before release.
            self.model.eq_solimp[:, :3] = [0.99, 0.9999, 0.001]
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
        self.render_size = None
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
        self.expert_status = "reset"
        self.cut = np.zeros(3, dtype=bool)
        self.protected_touched = set()
        self.terrain_contact = False
        self.action_clips = 0
        self.events = []
        self.target_id = int(rng.integers(3))
        self.stems = np.zeros((3, 3))
        self.roots = np.zeros((3, 3))
        rotation = Rotation.from_euler("z", self.azimuth, degrees=True) * Rotation.from_euler("y", self.tilt, degrees=True)
        self.stem_direction = rotation.apply([0., 0., 1.])
        quat = rotation.as_quat()[[3, 0, 1, 2]]
        for i in range(3):
            x = rng.uniform(0.52, 0.58) + (0.035 if ood else 0)
            y = (i - 1) * 0.10 + rng.uniform(-0.015, 0.015)
            height = rng.uniform(0.13, 0.17) + (0.025 if ood else 0)
            self.stems[i] = [x, y, height]
            length = (height - 0.03) / self.stem_direction[2]
            self.roots[i] = self.stems[i] - length * self.stem_direction
            body = self.model.body(f"root{i}")
            body.pos[:] = self.roots[i]
            body.quat[:] = quat
            geom = self.model.geom(f"stump{i}")
            geom.pos[:] = [0, 0, length / 2]
            geom.size[1] = length / 2
            q = self.upper_q[i]
            self.data.qpos[q:q + 7] = [x, y, height, *quat]
            self.model.body(f"upper{i}").pos[:] = [x, y, height]
            self.model.body(f"upper{i}").quat[:] = quat
            eq = self.eq_ids[i]
            self.model.eq_data[eq, 3:10] = [0, 0, length, 1, 0, 0, 0]
            self.data.eq_active[eq] = True
            target = i == self.target_id
            self.model.geom(f"marker{i}").pos[:] = self.roots[i] + [0, 0, 0.003]
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
        base = np.r_[(self.target - self.tip) * 10, self.tip,
                     self.data.qpos[self.scissor_q] / OPEN_ANGLE,
                     float(self.cut[self.target_id]), self.data.qpos[self.arm_q]].astype(np.float32)
        if self.observation_schema == "state15_v1":
            return base
        segments = [point for i in range(3) if i != self.target_id
                    for point in (self.roots[i], self.stems[i] + self.upper_length * self.stem_direction)]
        return np.r_[base, self.stem_direction, ((np.asarray(segments) - self.tip) * 10).ravel()].astype(np.float32)

    def expert(self):
        if self.teacher == "waypoint_v2":
            from expert import waypoint_action
            return waypoint_action(self)
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
            if self.geometry == "stem_segment_v1":
                start = rot.T @ (self.roots[i] - pos)
                end = rot.T @ (stem + self.upper_length * self.stem_direction - pos)
                hit = segment_cut_intersection(start, end, local, self.previous_angle, angle)
                cutting = hit is not None
            else:
                cutting = cutting_intersection(local, self.previous_angle, angle)
            if not self.cut[i] and cutting:
                self.cut[i] = True
                self.data.eq_active[self.eq_ids[i]] = False
                event = {"time_s": float(self.data.time), "plant": i,
                         "target": i == self.target_id, "height_error_m": abs(float(local[2]))}
                if self.geometry == "stem_segment_v1":
                    event.update(intersection_xyz=(pos + rot @ hit).tolist(),
                                 seam_distance_m=float(np.linalg.norm(hit - local)))
                self.events.append(event)
            # Conservative cutting-envelope intrusion, not a measured force.
            if i != self.target_id:
                if self.geometry == "stem_segment_v1":
                    touched = segment_protected_intersection(start, end, angle)
                else:
                    touched = (0.02 < local[0] < 0.155 and abs(local[2]) < 0.018
                               and -0.01 < local[1] < local[0] * np.sin(max(angle, 0)) + 0.01)
                if touched:
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

    def configuration(self):
        return {"scene": self.scene, "plant": self.plant, "tilt_degrees": self.tilt,
                "azimuth_degrees": self.azimuth, "geometry": self.geometry,
                "background_collision": False, "teacher": self.teacher,
                "observation_schema": self.observation_schema}

    def render(self, width=960, height=640, closeup=False, camera=None):
        name = camera or ("closeup" if closeup else "overview")
        lookat, distance, azimuth, elevation = camera_settings(name, self.target)
        if self.renderer is not None and self.render_size != (width, height):
            self.close()
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            self.render_size = (width, height)
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
