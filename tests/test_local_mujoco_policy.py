"""Check the saved G1 observation contract independently of simulator ordering."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def replay(monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("replay_g1_policy_mujoco")
    return module.PolicyReplay(module.DEFAULT_CHECKPOINT, [0.3, 0, 0])


def test_saved_actor_observation_layout_at_initial_pose(replay):
    observation = replay.observation()
    assert observation.shape == (100,)
    np.testing.assert_allclose(observation[:33], 0, atol=1e-7)
    np.testing.assert_allclose(observation[33:35], [0.3, 0], atol=1e-7)
    np.testing.assert_allclose(observation[35:37], [1, -1], atol=1e-7)
    np.testing.assert_allclose(observation[37:95], 0, atol=1e-7)
    np.testing.assert_allclose(observation[95:98], [0, 0, -1], atol=1e-7)
    np.testing.assert_allclose(observation[98:], 0, atol=1e-7)


def test_standing_phase_and_reset_clear_previous_actions(replay):
    replay.velocity[:] = 0
    replay.last_action[:] = 0.75
    replay.episode_steps = 12
    observation = replay.observation()
    np.testing.assert_allclose(observation[:29], 0.75)
    np.testing.assert_allclose(observation[35:37], [-1, -1])
    np.testing.assert_allclose(observation[98:], 0, atol=1e-7)
    replay.reset()
    np.testing.assert_array_equal(replay.last_action, 0)
    np.testing.assert_allclose(replay.data.qpos[replay.qpos_ids], replay.default_pos)


def test_pd_physics_steps_preserve_clock_and_effort_limits(replay):
    replay.step()
    assert replay.data.time == pytest.approx(0.02)
    assert replay.total_steps == 1
    assert np.isfinite(replay.data.qpos).all()
    assert np.any(np.abs(replay.last_action) > 1e-6)
    assert np.all(np.abs(replay.data.ctrl[replay.actuator_ids]) <= replay.torque_limits)
