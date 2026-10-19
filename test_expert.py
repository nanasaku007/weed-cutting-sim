from types import SimpleNamespace
import json

import numpy as np
import pytest

from environment import CuttingEnv, OBS_V2_NAMES, segment_protected_intersection
from expert import TIP_OFFSET, blocked_positions
from stage3 import episode
from run import TreeBC, train


def test_planner_envelope_matches_event_geometry():
    rng = np.random.default_rng(28)
    for angle in (0., 0.2, 0.65):
        positions = rng.uniform([0.3, -0.2, 0.1], [0.65, 0.2, 0.4], (120, 3))
        roots = np.array([[0.55, -0.1, 0.03], [0.57, 0.1, 0.03]])
        tops = roots + [[0.06, 0.02, 0.3], [-0.06, -0.04, 0.35]]
        expected = [any(segment_protected_intersection(a - p + TIP_OFFSET, b - p + TIP_OFFSET, angle)
                        for a, b in zip(roots, tops)) for p in positions]
        np.testing.assert_array_equal(blocked_positions(positions, roots, tops, angle, radius=0.003), expected)


@pytest.mark.parametrize("seed,tilt,azimuth,recovery", [
    (41001, 0, 0, False), (42006, 10, 180, False),
    (42006, 20, 180, False), (42000, 0, 0, True),
    (42001, 20, 90, True), (42003, 10, 270, True),
])
def test_teacher_avoids_stems_and_returns(seed, tilt, azimuth, recovery):
    env = CuttingEnv(plant="broad", tilt=tilt, azimuth=azimuth, geometry="stem_segment_v1",
                     teacher="waypoint_v2", observation_schema="state30_v2")
    try:
        result, frames = episode(env, seed, recovery, record=True)
        assert result["success"], result
        assert result["protected_intrusions"] == result["wrong_cuts"] == 0
        assert len(frames[0][0]) == len(OBS_V2_NAMES) == 30
        if recovery:
            assert frames[0][0][6] == 0  # starts closed
            assert result["reopened_before_cut"]
        env.reset(seed)
        first = env.expert()
        np.testing.assert_array_equal(first, env.expert())  # no hidden route/phase state
    finally:
        env.close()


def test_observation_exposes_protected_geometry():
    env = CuttingEnv(observation_schema="state30_v2")
    try:
        before = env.reset(5)
        other = (env.target_id + 1) % 3
        env.stems[other, 1] += 0.03
        after = env.observe()
        np.testing.assert_array_equal(before[:15], after[:15])
        assert not np.array_equal(before[18:], after[18:])
    finally:
        env.close()


def test_numeric_checkpoint_schema_and_prescribed_split(tmp_path):
    rng = np.random.default_rng(5)
    data, checkpoint = tmp_path / "data.npz", tmp_path / "bc.npz"
    np.savez_compressed(data, observation=rng.normal(size=(60, 30)), action=rng.uniform(size=(60, 4)),
                        episode_id=np.repeat(np.arange(10), 6), observation_schema="state30_v2",
                        validation_episode_ids=[1, 6])
    train(SimpleNamespace(data=data, out=checkpoint, seed=42))
    model = TreeBC(checkpoint)
    assert model.predict(np.zeros(30)).shape == (4,)
    with pytest.raises(ValueError, match="state30_v2"):
        model.predict(np.zeros(15))
    report = json.loads(checkpoint.with_suffix(".json").read_text())
    assert set(report["validation_episode_ids"]) == {1, 6}
    assert set(report["train_episode_ids"]) == set(range(10)) - {1, 6}
    # Legacy numeric checkpoints have no schema field and continue to require 15.
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, left=[-1], right=[-1], feature=[-2], threshold=[0.], values=np.zeros((1, 4)), starts=[0])
    old = TreeBC(legacy)
    assert old.predict(np.zeros(15)).shape == (4,)
    with pytest.raises(ValueError, match="state15_v1"):
        old.predict(np.zeros(30))
