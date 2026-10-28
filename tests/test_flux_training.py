"""Verify training fails closed before GPU work and preserves its wall-time limit."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def training(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("train_flux")
    monkeypatch.setattr(module, "repository_commit", lambda: "test-commit")
    return module


def test_invalid_scale_cannot_start_bootstrap(training, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_flux.py", "--num-envs", "0"])
    monkeypatch.setattr(training.bootstrap, "prepare_environment", lambda: pytest.fail("Invalid job reached bootstrap"))
    with pytest.raises(ValueError, match="Training requires"):
        training.main()


def test_install_failure_does_not_start_gpu(training, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_flux.py"])
    monkeypatch.setattr(training.bootstrap, "prepare_environment", dict)

    def fail_install(env, timeout):
        raise subprocess.CalledProcessError(1, "pip")

    monkeypatch.setattr(training.bootstrap, "install_training_package", fail_install)
    monkeypatch.setattr(training.bootstrap, "check_gpu", lambda _env: pytest.fail("Failed install reached GPU"))
    with pytest.raises(subprocess.CalledProcessError):
        training.main()


def test_training_timeout_is_enforced_without_retry(training, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_flux.py", "--training-timeout", "21600"])
    monkeypatch.setattr(training.bootstrap, "prepare_environment", dict)
    monkeypatch.setattr(training.bootstrap, "install_training_package", lambda _env, _timeout: None)
    monkeypatch.setattr(training.bootstrap, "check_gpu", lambda _env: None)
    attempted = []

    def timeout_training(command, *, check, timeout, env):
        attempted.append(command)
        assert check and timeout == 21600
        assert "--training.num-envs=2048" in command
        assert "--algo.config.num-learning-iterations=5000" in command
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(training.subprocess, "run", timeout_training)
    with pytest.raises(subprocess.TimeoutExpired):
        training.main()
    assert len(attempted) == 1
