"""Protect native evaluation identity and pre-reset diagnostic accounting."""

import importlib
from pathlib import Path
from types import SimpleNamespace as Namespace

import pytest
import torch


@pytest.fixture
def evaluation(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "src/holosoma"))
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return importlib.import_module("eval_flux_box")


def test_real_checkpoint_keeps_native_physics_and_termination(evaluation):
    from holosoma.config_types.experiment import ExperimentConfig  # noqa: PLC0415

    checkpoint = evaluation.ROOT / "czy/data/exp_2.1/model_07999.pt"
    saved = ExperimentConfig(**torch.load(checkpoint, map_location="cpu", weights_only=False)["experiment_config"])
    config = evaluation.evaluation_config(saved, {"num_envs": 32, "project": "test-native-eval"})
    for name in ("robot", "scene", "command", "randomization", "termination", "reward"):
        assert getattr(config, name) == getattr(saved, name)
    assert config.training.num_envs == 32
    assert config.simulator.config.sim.max_episode_length_s == 10
    assert config.algo.config.empirical_normalization and not config.algo.config.load_optimizer
    assert not config.logger.video.enabled and not config.logger.headless_recording
    assert not any(group.enable_noise for group in config.observation.groups.values())


def test_terminal_snapshot_counts_overlapping_causes_once_and_enforces_env_quota(evaluation):
    masks = {
        name: torch.zeros(2, dtype=torch.bool)
        for name in (
            "bad_ref_pos",
            "bad_ref_ori",
            "bad_motion_body_pos",
            "bad_object_pos",
            "bad_object_ori",
        )
    }
    timeout = torch.zeros(2, dtype=torch.bool)
    motion = Namespace(
        body_pos_relative_w=torch.zeros(2, 1, 3),
        robot_body_pos_w=torch.zeros(2, 1, 3),
        simulator_object_pos_w=torch.zeros(2, 3),
        object_pos_w=torch.zeros(2, 3),
        object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        simulator_object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        robot_root_pos_w=torch.zeros(2, 3),
        time_steps=torch.tensor([324, 0]),
        motion=Namespace(motion_end_idx=torch.tensor([325])),
        motion_ids=torch.zeros(2, dtype=torch.long),
    )
    bad = Namespace(
        bad_motion_body_pos_body_indexes=torch.tensor([0]),
        bad_motion_body_pos_body_names=["wrist"],
        bad_motion_body_pos_threshold=0.25,
    )
    for name, value in masks.items():
        setattr(bad, name, lambda _motion, value=value: value)
    manager = Namespace(
        _term_instances={"bad_tracking": bad},
        check=lambda: (
            torch.stack(list(masks.values())).any(dim=0),
            timeout.clone(),
        ),
    )
    env = Namespace(
        num_envs=2,
        device="cpu",
        dt=0.02,
        termination_manager=manager,
        command_manager=Namespace(get_state=lambda _name: motion),
        simulator=Namespace(scene=Namespace(env_origins=torch.zeros(2, 3))),
        episode_length_buf=torch.tensor([325, 1]),
    )
    observer = evaluation.NativeObserver(env, 1, lambda a, _b: torch.zeros(a.shape[0]))
    observer.active = True
    manager.check()  # Env 0 completes a clip, without terminating its episode.
    masks["bad_ref_pos"][0] = True
    masks["bad_object_pos"][0] = True
    timeout[1] = True
    env.episode_length_buf[:] = torch.tensor([350, 500])
    manager.check()  # Capture state before the simulator resets it.
    motion.simulator_object_pos_w[:] = 100
    manager.check()  # Fast environments must not contribute extra episodes.
    result = observer.result()["summary"]
    assert result["completed_episodes"] == 2 and result["first_clip_success_rate"] == 0.5
    assert result["termination_counts_overlapping"] == {"bad_ref_pos": 1, "bad_object_pos": 1, "timeout": 1}
    assert observer.episodes[0]["object_position_error_m"] == 0
    assert observer.episodes[0]["duration_s"] == 7


def test_incomplete_evaluation_is_rejected(evaluation):
    with pytest.raises(RuntimeError, match="0/128"):
        evaluation.summarize([], 128)


def test_resumed_box_video_uses_final_model_and_native_timeout(evaluation, monkeypatch, tmp_path):
    import replay_flux  # noqa: PLC0415
    from holosoma.config_types.experiment import ExperimentConfig  # noqa: PLC0415

    path = evaluation.ROOT / "czy/data/exp_2.2/model_13999.pt"
    saved = ExperimentConfig(**torch.load(path, map_location="cpu", weights_only=False)["experiment_config"])
    config = replay_flux.replay_configuration(
        saved,
        {
            "checkpoint": str(path),
            "box_task": True,
            "seconds": 30,
            "segments": str(tmp_path),
            "project": "test-resumed-box-video",
        },
    )
    assert config.training.num_envs == 1 and config.training.max_eval_steps == 1500
    assert config.simulator.config.sim.max_episode_length_s == 10
    assert config.logger.video.enabled and not config.algo.config.load_optimizer
    assert config.logger.video.width == 1280 and config.logger.video.height == 720
    assert all(not group.enable_noise for group in config.observation.groups.values())
