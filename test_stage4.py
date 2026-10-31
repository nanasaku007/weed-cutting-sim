import numpy as np
import pytest

from environment import CuttingEnv
from stage3 import episode
from stage4 import rollout, summarize


@pytest.mark.parametrize("recovery", [False, True])
def test_teacher_rollout_matches_stage3(recovery):
    env = CuttingEnv(plant="broad", tilt=10, azimuth=180, geometry="stem_segment_v1",
                     teacher="waypoint_v2", observation_schema="state30_v2")
    try:
        previous, frames = episode(env, 42006, recovery, record=True)
        current, trace = rollout(env, 42006, recovery=recovery)
        for key in previous:
            assert previous[key] == current[key], key
        np.testing.assert_array_equal(trace["action"], [f[1] for f in frames])
    finally:
        env.close()


def test_bc_rollout_never_calls_teacher(monkeypatch):
    class ConstantPolicy:
        observation_schema = "state30_v2"

        def predict(self, obs):
            assert obs.shape == (30,)
            return np.array([0., 0., 0., 1.])

    env = CuttingEnv(max_steps=3, observation_schema="state30_v2")
    def forbidden():
        pytest.fail("BC called the teacher")
    monkeypatch.setattr(env, "expert", forbidden)
    try:
        result, trace = rollout(env, 42006, ConstantPolicy())
        assert result["outcome"] == "timeout"
        np.testing.assert_array_equal(trace["action"], [[0, 0, 0, 1]] * 3)
        assert result["phases"] == {"bc": 3}
        ConstantPolicy.observation_schema = "state15_v1"
        with pytest.raises(ValueError, match="schemas differ"):
            rollout(env, 42006, ConstantPolicy())
    finally:
        env.close()


def test_initialization_failure_stays_in_denominator(monkeypatch):
    def invalid(*args):
        raise ValueError("recovery_start_blocked")
    monkeypatch.setattr("stage4.initialize", invalid)
    env = CuttingEnv(observation_schema="state30_v2")
    try:
        row, trace = rollout(env, 42000, recovery=True)
        summary = summarize([row])
        assert trace == {}
        assert summary["count"] == summary["initialization_failures"] == 1
        assert summary["success_rate"] == 0
        assert summary["mean_sim_seconds"] is None
    finally:
        env.close()


def test_summary_counts_overlapping_safety_events_and_cut_timeouts():
    common = dict(success=False, target_cut=True, wrong_cuts=0, protected_intrusions=0,
                  tool_terrain_contact=False, steps=180, sim_seconds=9.)
    rows = [dict(common, outcome="timeout"),
            dict(common, outcome="wrong_cut", wrong_cuts=1, protected_intrusions=1,
                 steps=20, sim_seconds=1.),
            dict(common, outcome="timeout", target_cut=False)]
    result = summarize(rows)
    assert result["count"] == 3
    assert result["wrong_cut_episodes"] == result["protected_intrusion_episodes"] == 1
    assert result["intrusion_after_cut"] == 1
    assert result["time_limit_episodes"] == 2
    assert result["timeout_before_cut"] == result["timeout_after_cut"] == 1
    assert result["success_mean_sim_seconds"] is None


def test_partial_evaluation_cannot_be_reported_as_final():
    from stage4_report import evaluation_summary
    with pytest.raises(ValueError, match="incomplete"):
        evaluation_summary({"complete": False, "conditions": []})
