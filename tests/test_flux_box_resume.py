"""Guard continuation state, exact update numbering, and the capacity gate."""

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def resume(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "src/holosoma"))
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return importlib.import_module("resume_flux_box")


def test_real_source_preserves_task_and_uses_native_scale(resume):
    source = resume.load_source()
    cfg = resume.resume_config(source, "resume-test", 6000)
    actual = cfg.to_serializable_dict()
    for name in ("robot", "scene", "command", "action", "observation", "reward", "termination", "randomization"):
        assert actual[name] == source["experiment_config"][name]
    assert cfg.training.num_envs == 4096 and cfg.training.seed == 42
    assert cfg.training.checkpoint == str(resume.SOURCE)
    assert cfg.algo.config.num_learning_iterations == 6000 and cfg.algo.config.load_optimizer
    assert cfg.algo.config.save_interval == 100
    assert resume.resume_config(source, "custom", 6000, save_interval=200).algo.config.save_interval == 200
    assert resume.resume_config(source, "capacity", 2, capacity=True).algo.config.save_interval == 1
    resume.box.check_task_config(actual, allow_resume=True)
    with pytest.raises(ValueError, match="from scratch"):
        resume.box.check_task_config(actual)


class TinyRunner:
    def __init__(self):
        self.actor = torch.nn.Linear(2, 1)
        self.critic = torch.nn.Linear(2, 1)
        self.actor_obs_normalizer = torch.nn.Linear(1, 1)
        self.critic_obs_normalizer = torch.nn.Linear(1, 1)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=3.375e-5)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=3.375e-5)
        for module, opt in [(self.actor, self.actor_optimizer), (self.critic, self.critic_optimizer)]:
            module(torch.ones(1, 2)).sum().backward()
            opt.step()
        self.config = SimpleNamespace(num_learning_iterations=6000)
        self.logging_helper = SimpleNamespace(num_learning_iterations=6000)
        self.skip_optimizer = False

    def load(self, _path):
        for key, name in [
            ("actor_model_state_dict", "actor"),
            ("critic_model_state_dict", "critic"),
            ("actor_obs_normalizer_state_dict", "actor_obs_normalizer"),
            ("critic_obs_normalizer_state_dict", "critic_obs_normalizer"),
        ]:
            getattr(self, name).load_state_dict(self.source[key])
        if not self.skip_optimizer:
            self.actor_optimizer.load_state_dict(self.source["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(self.source["critic_optimizer_state_dict"])
        self.actor_learning_rate = self.actor_optimizer.param_groups[0]["lr"]
        self.critic_learning_rate = self.critic_optimizer.param_groups[0]["lr"]
        self.current_learning_iteration = self.source["iter"]


def test_restore_proves_optimizer_state_and_does_not_repeat_last_update(resume):
    runner = TinyRunner()
    source = {"iter": 7999}
    for key, name in [
        ("actor_model_state_dict", "actor"),
        ("critic_model_state_dict", "critic"),
        ("actor_obs_normalizer_state_dict", "actor_obs_normalizer"),
        ("critic_obs_normalizer_state_dict", "critic_obs_normalizer"),
        ("actor_optimizer_state_dict", "actor_optimizer"),
        ("critic_optimizer_state_dict", "critic_optimizer"),
    ]:
        source[key] = copy.deepcopy(getattr(runner, name).state_dict())
    runner.source = source
    result = resume.restore(runner, source, "trusted.pt")
    assert result["next_iteration"] == 8000 and result["expected_final_iteration"] == 13999
    assert len(range(runner.current_learning_iteration, runner.current_learning_iteration + 6000)) == 6000
    runner.skip_optimizer = True
    runner.actor_optimizer.state.clear()
    with pytest.raises(RuntimeError, match="actor_optimizer_state_dict"):
        resume.restore(runner, source, "trusted.pt")


def test_capacity_failure_cannot_start_formal_continuation(resume, monkeypatch, tmp_path):
    monkeypatch.setattr(resume, "ROOT", tmp_path)
    monkeypatch.setattr(resume.sys, "argv", ["resume_flux_box.py"])
    monkeypatch.setattr(resume, "repository_commit", lambda: "test")
    monkeypatch.setattr(resume.box.bootstrap, "prepare_environment", dict)
    monkeypatch.setattr(resume.box.bootstrap, "install_training_package", lambda *_: None)
    monkeypatch.setattr(resume.box.bootstrap, "check_gpu", lambda *_: None)
    calls = []

    def fail_capacity(_root, _project, iterations, capacity, _env, _timeout, **_kwargs):
        calls.append((iterations, capacity))
        raise RuntimeError("capacity failed")

    monkeypatch.setattr(resume, "stage", fail_capacity)
    with pytest.raises(RuntimeError, match="capacity failed"):
        resume.main()
    assert calls == [(2, True)]


def test_authorized_skip_reuses_capacity_and_passes_formal_save_interval(resume, monkeypatch, tmp_path):
    monkeypatch.setattr(resume, "ROOT", tmp_path)
    monkeypatch.setattr(resume.sys, "argv", ["resume_flux_box.py", "--skip-capacity-test", "--save-interval", "100"])
    monkeypatch.setattr(resume, "repository_commit", lambda: "test")
    monkeypatch.setattr(resume.box.bootstrap, "prepare_environment", dict)
    monkeypatch.setattr(resume.box.bootstrap, "install_training_package", lambda *_: None)
    monkeypatch.setattr(resume.box.bootstrap, "check_gpu", lambda *_: None)
    monkeypatch.setattr(resume.time, "sleep", lambda *_: None)
    calls = []
    monkeypatch.setattr(resume, "stage", lambda *args, **kwargs: calls.append((args[2:4], kwargs)))
    resume.main()
    assert len(calls) == 1 and calls[0][0] == (6000, False)
    assert calls[0][1]["save_interval"] == 100 and calls[0][1]["source_iteration"] == 7999


def test_latest_box_checkpoint_can_resume_for_10000_updates(resume):
    path = resume.ROOT / "czy/data/exp_2.2/model_13999.pt"
    digest = "7546092cda21371a8c37a6fac3eb92952a560b9e0bf2484c256bec3a92d05120"
    source = resume.load_source(path, digest, 13999)
    cfg = resume.resume_config(source, "exp_2.3", 10000, checkpoint=path)
    assert cfg.training.checkpoint == str(path) and cfg.training.num_envs == 4096
    assert cfg.algo.config.num_learning_iterations == 10000 and cfg.algo.config.save_interval == 100
    assert cfg.algo.config.load_optimizer
    assert int(source["iter"]) + 1 == 14000 and int(source["iter"]) + 10000 == 23999
    with pytest.raises(AssertionError):
        resume.load_source(path, digest, 7999)


def test_cli_propagates_latest_checkpoint_and_exact_iteration_budget(resume, monkeypatch, tmp_path):
    path = resume.ROOT / "czy/data/exp_2.2/model_13999.pt"
    digest = "7546092cda21371a8c37a6fac3eb92952a560b9e0bf2484c256bec3a92d05120"
    monkeypatch.setattr(resume, "ROOT", tmp_path)
    monkeypatch.setattr(
        resume.sys,
        "argv",
        [
            "resume_flux_box.py",
            "--experiment",
            "exp_2.3",
            "--checkpoint",
            str(path),
            "--checkpoint-sha256",
            digest,
            "--source-iteration",
            "13999",
            "--additional-iterations",
            "10000",
            "--skip-capacity-test",
        ],
    )
    monkeypatch.setattr(resume, "repository_commit", lambda: "test")
    monkeypatch.setattr(resume.box.bootstrap, "prepare_environment", dict)
    monkeypatch.setattr(resume.box.bootstrap, "install_training_package", lambda *_: None)
    monkeypatch.setattr(resume.box.bootstrap, "check_gpu", lambda *_: None)
    monkeypatch.setattr(resume.time, "sleep", lambda *_: None)
    calls = []
    monkeypatch.setattr(resume, "stage", lambda *args, **kwargs: calls.append((args, kwargs)))
    resume.main()
    assert len(calls) == 1 and calls[0][0][2:4] == (10000, False)
    assert calls[0][1]["checkpoint"] == path and calls[0][1]["source_iteration"] == 13999
    assert calls[0][1]["checkpoint_sha256"] == digest and calls[0][1]["save_interval"] == 100
