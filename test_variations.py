"""Geometry and dynamics checks, separate from policy success evaluation."""
import mujoco
import numpy as np
import pytest

from environment import (CuttingEnv, segment_cut_intersection,
                         segment_protected_intersection)


@pytest.mark.parametrize("tilt", [0, 10, 20])
@pytest.mark.parametrize("azimuth", [0, 90, 210])
def test_plane_crossing_at_a_tilt(tilt, azimuth):
    theta, phi = np.deg2rad([tilt, azimuth])
    direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    seam = np.array([0.085, 0.004, 0.003])
    start, end = seam - 0.13 * direction, seam + 0.14 * direction
    hit = segment_cut_intersection(start, end, seam, 0.2, 0.05)
    np.testing.assert_allclose(hit, seam - direction * seam[2] / direction[2], atol=1e-12)
    for shift in ([0.3, 0, 0], [0, 0.2, 0], [0, 0, 0.04]):
        assert segment_cut_intersection(start + shift, end + shift, seam + shift, 0.2, 0.05) is None
    assert segment_cut_intersection(start, end, seam, 0.05, 0.2) is None
    assert segment_cut_intersection(start, end, seam, 0.05, 0.03) is None


def test_protection_detects_stem_even_when_seam_is_above_blades():
    assert segment_protected_intersection(np.array([0.085, 0.004, -0.08]),
                                          np.array([0.085, 0.004, 0.18]), 0.5)
    assert not segment_protected_intersection(np.array([0.3, 0.004, -0.08]),
                                              np.array([0.3, 0.004, 0.18]), 0.5)
    # A tilted segment crossing the envelope with both endpoints outside it.
    assert segment_protected_intersection(np.array([0, 0.004, -0.05]),
                                          np.array([0.20, 0.004, 0.05]), 0.5)


@pytest.mark.parametrize("start,end", [
    ([0.05, 0.004, 0.01], [0.12, 0.004, 0.01]),  # parallel
    ([0.085, 0.004, 0.01], [0.085, 0.004, 0.15]),  # entirely above
    ([0.085, 0.004, -0.15], [0.085, 0.004, -0.01]),  # entirely below
])
def test_segment_must_actually_cross_cutting_plane(start, end):
    assert segment_cut_intersection(np.array(start), np.array(end), np.array(start), 0.2, 0.05) is None


@pytest.mark.parametrize("plant", ["slender", "broad", "tall"])
@pytest.mark.parametrize("tilt", [0, 10, 20])
def test_nine_plants_weld_release_fall_and_reset(plant, tilt):
    env = CuttingEnv(plant=plant, tilt=tilt, azimuth=70, geometry="stem_segment_v1")
    try:
        initial = env.reset(0)
        np.testing.assert_allclose(env.roots[:, 2], 0.03)
        # Move directly to the seam to isolate geometry/dynamics from expert paths.
        q, residual = env.solve_ik(env.target, env.rest_q, iterations=150)
        assert residual < 0.001
        env.data.qpos[env.arm_q] = q
        env.data.ctrl[:7] = q
        mujoco.mj_forward(env.model, env.data)
        for _ in range(5):
            env.step([0, 0, 0, 1])
        for i in range(3):
            np.testing.assert_allclose(env.data.body(f"upper{i}").xpos, env.stems[i], atol=0.001)
            np.testing.assert_allclose(env.data.body(f"upper{i}").xmat.reshape(3, 3)[:, 2],
                                       env.stem_direction, atol=0.001)
        assert not env.cut.any()
        for _ in range(12):
            env.step([0, 0, 0, 0])
        assert env.cut[env.target_id]
        assert not env.data.eq_active[env.eq_ids[env.target_id]]
        assert env.data.body(f"upper{env.target_id}").xpos[2] < env.target[2] - 0.035
        assert env.data.eq_active[env.eq_ids].sum() == 2
        for _ in range(35):
            env.step([0, 0, 0, 0])
        # Falling must end in contact, not continue through the floor.
        # A tilted plant can roll off the soil bed onto the z=0 floor.
        assert env.data.body(f"upper{env.target_id}").xpos[2] > -0.005
        contacts = [c for c in env.data.contact
                    if env.model.geom(f"stem{env.target_id}").id in (c.geom1, c.geom2)]
        assert contacts
        assert min(c.dist for c in contacts) > -0.001
        np.testing.assert_allclose(env.reset(0), initial)
        assert env.data.eq_active[env.eq_ids].all()
        assert all(env.model.body_contype[env.model.body(f"upper{i}").id] == 4 for i in range(3))
        # A real closure at home must not detach any stem.
        for _ in range(12):
            env.step([0, 0, 0, 0])
        assert not env.cut.any()
    finally:
        env.close()


def test_background_does_not_change_physics_or_observations():
    reference = CuttingEnv()
    initial = reference.reset(8)
    actions, observations = [], []
    done = False
    while not done:
        action = reference.expert()
        obs, done, expected = reference.step(action)
        actions.append(action)
        observations.append(obs)
    for scene in ("curb", "tree", "river"):
        env = CuttingEnv(scene=scene)
        np.testing.assert_array_equal(env.reset(8), initial)
        for geom_id in range(env.model.ngeom):
            if (env.model.geom(geom_id).name or "").startswith("decor_"):
                assert env.model.geom_contype[geom_id] == env.model.geom_conaffinity[geom_id] == 0
        for action, obs in zip(actions, observations):
            current, _, actual = env.step(action)
            np.testing.assert_array_equal(current, obs)
        assert actual == expected
        env.close()
    reference.close()


def test_tilt_requires_versioned_geometry():
    with pytest.raises(ValueError, match="Tilt requires"):
        CuttingEnv(tilt=10)
    with pytest.raises(ValueError, match="tilt must"):
        CuttingEnv(tilt=float("nan"))
