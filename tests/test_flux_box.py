"""Protect the object-task contract and the smoke gate before paid training."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def box(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("train_flux_box")


def test_command_selects_object_tracking_and_never_resumes_locomotion(box):
    command = box.command(1024, 8000, 200, "flux-exp_2-box-test")
    assert "exp:g1-29dof-wbt-w-object" in command
    assert "exp:g1-29dof" not in command
    assert not any("checkpoint" in item for item in command)
    assert "--training.num-envs=1024" in command


def test_robot_only_motion_is_rejected_before_gpu(box):
    cfg = {
        "env_class": "holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
        "command": {
            "setup_terms": {"motion_command": {"params": {"motion_config": {"motion_file": "robot_only.npz"}}}}
        },
    }
    with pytest.raises(ValueError, match="object motion"):
        box.check_task_config(cfg)


def test_smoke_failure_cannot_start_formal_training(box, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_flux_box.py"])
    monkeypatch.setattr(box, "repository_commit", lambda: "test-commit")
    monkeypatch.setattr(box.bootstrap, "prepare_environment", dict)
    monkeypatch.setattr(box.bootstrap, "install_training_package", lambda *_: None)
    monkeypatch.setattr(box.bootstrap, "check_gpu", lambda *_: None)
    monkeypatch.setattr(box, "fresh", lambda *_: None)
    monkeypatch.setattr(box, "verify_fresh", lambda *_: pytest.fail("Failed smoke reached acceptance"))
    attempted = []

    def fail_smoke(command, *, check, timeout, env):
        attempted.append(command)
        assert check and timeout == 900
        assert "--training.num-envs=128" in command
        assert "--algo.config.num-learning-iterations=2" in command
        raise subprocess.CalledProcessError(1, "smoke")

    monkeypatch.setattr(box.subprocess, "run", fail_smoke)
    with pytest.raises(subprocess.CalledProcessError):
        box.main()
    assert len(attempted) == 1
