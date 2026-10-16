import numpy as np
import pytest
import mujoco

from environment import CUT_ANGLE, CuttingEnv, cutting_intersection


@pytest.mark.parametrize("local,before,after,expected", [
    ([0.085, 0.004, 0], 0.2, 0.05, True),
    ([0.085, 0.004, 0], 0.05, 0.2, False),  # opening must not cut
    ([0.085, 0.004, 0], 0.05, 0.03, False), # staying closed must not cut
    ([0.25, 0.004, 0], 0.2, 0.05, False),  # beyond blade tip
    ([0.085, 0.05, 0], 0.2, 0.05, False), # beside blades
    ([0.085, 0.004, 0.04], 0.2, 0.05, False), # above cut plane
])
def test_cut_requires_closing_edge_and_geometry(local, before, after, expected):
    assert cutting_intersection(local, before, after) == expected


@pytest.fixture(scope="module")
def env():
    e = CuttingEnv()
    yield e
    e.close()


def test_expert_closed_loop_and_reproducible_reset(env):
    initial = env.reset(seed=0)
    target = env.target.copy()
    done = False
    while not done:
        _, done, result = env.step(env.expert())
    assert result["success"], result
    assert not env.data.eq_active[env.eq_ids[env.target_id]]
    np.testing.assert_allclose(env.reset(seed=0), initial)
    np.testing.assert_allclose(env.target, target)
    assert env.data.eq_active[env.eq_ids].all()
    assert not env.cut.any()
    assert env.events == []


def test_invalid_action_and_output_bounds(env):
    env.reset(seed=5)
    with pytest.raises(ValueError):
        env.step([0, np.nan, 0, 1])
    with pytest.raises(ValueError):
        env.step([0, 1])
    env.step([200, -200, 200, 5])
    assert env.action_clips == 1
    assert np.all(env.data.ctrl[:7] >= env.limits[:, 0])
    assert np.all(env.data.ctrl[:7] <= env.limits[:, 1])


def test_uncut_stems_remain_welded_at_their_generated_positions(env):
    env.reset(seed=7)
    for _ in range(8):
        env.step([0, 0, 0, 1])
    for i in range(3):
        np.testing.assert_allclose(env.data.body(f"upper{i}").xpos, env.stems[i], atol=0.001)
    assert not env.cut.any()


def test_wrong_plant_cut_is_not_success(env):
    env.reset(seed=1)
    wrong_id = (env.target_id + 1) % 3
    rot = env.data.xmat[env.tool_id].reshape(3, 3)
    env.stems[wrong_id] = env.data.xpos[env.tool_id] + rot @ np.array([0.085, 0.004, 0])
    env.previous_angle = CUT_ANGLE + 0.1
    env.data.qpos[env.scissor_q] = CUT_ANGLE - 0.02
    mujoco.mj_forward(env.model, env.data)
    env._events()
    assert env.cut[wrong_id]
    assert not env.result()["success"]
    assert env.result()["wrong_cuts"] == 1
