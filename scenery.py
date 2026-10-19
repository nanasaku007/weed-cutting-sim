"""Deterministic, non-colliding scenery and named camera presets.

Background RNG is independent of episode RNG. Existing floor/soil/curbs remain
the physical terrain in every scene; all decor_* geoms are visual only.
"""
import xml.etree.ElementTree as ET

import numpy as np

SCENES = ("baseline", "curb", "tree", "river")
PLANTS = ("legacy", "slender", "broad", "tall")
CAMERAS = ("overview", "closeup", "top")


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def numbers(values):
    return " ".join(f"{v:.6g}" for v in values)


def decorate(root, scene):
    if scene == "baseline":
        return
    asset, world = root.find("asset"), root.find("worldbody")
    add(asset, "texture", name="decor_sky", type="skybox", builtin="gradient",
        rgb1="0.56 0.68 0.72", rgb2="0.84 0.86 0.79", width="512", height="3072")
    palette = {"asphalt": ("0.16 0.18 0.18", "0.29 0.30 0.29"),
               "earth": ("0.25 0.17 0.10", "0.38 0.29 0.18"),
               "grass": ("0.21 0.30 0.13", "0.34 0.39 0.20"),
               "concrete": ("0.47 0.48 0.44", "0.65 0.64 0.56")}
    for name, (color, fleck) in palette.items():
        add(asset, "texture", name=f"decor_{name}_tex", type="2d", builtin="flat",
            rgb1=color, rgb2=color, mark="random", markrgb=fleck,
            random="0.12", width="256", height="256")
        add(asset, "material", name=f"decor_{name}", texture=f"decor_{name}_tex",
            texrepeat="10 10", specular="0.05", shininess="0.05", reflectance="0")
    world.find("geom[@name='floor']").set("material", "decor_grass" if scene == "river" else "decor_asphalt")
    soil = world.find("geom[@name='soil']")
    soil.set("material", "decor_earth")
    soil.attrib.pop("rgba", None)
    for geom in world.findall("geom"):
        if geom.get("name", "").startswith("curb_"):
            geom.set("material", "decor_concrete")
            geom.attrib.pop("rgba", None)

    counter = 0

    def geom(kind, **attrs):
        nonlocal counter
        counter += 1
        return add(world, "geom", name=f"decor_{counter}", type=kind,
                   contype="0", conaffinity="0", mass="0", **attrs)

    rng = np.random.default_rng(981)
    # Keep the working strip clear; detail lives behind and beside the targets.
    for _ in range(65):
        x, y = rng.uniform(0.77, 1.35), rng.uniform(-0.75, 0.75)
        size = rng.uniform(0.004, 0.014)
        geom("ellipsoid", pos=numbers([x, y, size * 0.4]),
             size=numbers([size, size * 0.7, size * 0.4]), rgba="0.43 0.40 0.32 1")
    if scene == "curb":
        for y in (-0.21, 0.21):
            geom("box", pos=numbers([0.58, y, 0.0305]), size="0.155 0.055 0.001",
                 material="decor_concrete")
        # A narrow dark seam on top of the soil. All possible root positions fit.
        geom("box", pos="0.55 0 0.0303", size="0.035 0.15 0.0002", rgba="0.12 0.10 0.07 1")
        for y in (-0.6, 0.6):
            geom("box", pos=numbers([0.95, y, 0.045]), size="0.12 0.28 0.045", material="decor_concrete")
    elif scene == "tree":
        geom("cylinder", pos="0.95 0.33 0.48", size="0.085 0.48", rgba="0.30 0.21 0.13 1")
        for angle in np.linspace(0, 2 * np.pi, 7, endpoint=False):
            end = [0.95 + 0.23 * np.cos(angle), 0.33 + 0.23 * np.sin(angle), 0.012]
            geom("capsule", fromto=numbers([0.95, 0.33, 0.09, *end]), size="0.022", rgba="0.31 0.22 0.13 1")
        for _ in range(40):
            geom("ellipsoid", pos=numbers([rng.uniform(0.77, 1.3), rng.uniform(-0.6, 0.6), 0.008]),
                 size="0.025 0.012 0.002", euler=numbers([0, 0, rng.uniform(-3, 3)]),
                 rgba="0.48 0.28 0.10 1")
    else:
        geom("box", pos="1.5 0 -0.018", size="0.30 1.8 0.02", rgba="0.18 0.38 0.42 1")
        geom("box", pos="1.17 0 0.014", size="0.025 1.8 0.014", material="decor_concrete")
        for x in (1.3, 1.43, 1.65):
            geom("box", pos=numbers([x, 0, 0.003]), size="0.002 1.7 0.0003", rgba="0.39 0.57 0.56 1")
    if scene in ("tree", "river"):
        for _ in range(55 if scene == "river" else 24):
            x, y = rng.uniform(0.8, 1.10), rng.uniform(-0.8, 0.8)
            for offset in (-0.02, 0.01, 0.025):
                geom("capsule", fromto=numbers([x, y, 0.002, x + offset, y + offset, rng.uniform(0.06, 0.13)]),
                     size="0.002", rgba="0.31 0.43 0.16 1")


def camera_settings(name, target):
    if name == "overview":
        return [0.55, 0, 0.27], 1.85, 135, -30
    if name == "closeup":
        return [target[0] - 0.025, target[1], target[2] + 0.065], 0.62, 115, -23
    if name == "top":
        return [0.55, 0, 0.12], 0.82, 90, -89
    raise ValueError(f"Unknown camera {name!r}; choose from {CAMERAS}")
