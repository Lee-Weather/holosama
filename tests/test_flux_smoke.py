"""CPU checks for the cloud smoke test's artifact acceptance criteria."""

import importlib.util
import shutil
import urllib.error
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("flux_smoke", ROOT / "scripts/train_flux_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_smoke_refuses_large_training():
    with pytest.raises(ValueError, match="Smoke tests require"):
        smoke.training_command(4096, 25000, "test")


def test_index_selection_falls_back_after_dns_failure(monkeypatch):
    visited = []

    def open_index(url, timeout):
        visited.append(url)
        if len(visited) == 1:
            raise urllib.error.URLError("DNS unavailable")
        return nullcontext(SimpleNamespace(status=200))

    monkeypatch.setattr(smoke.urllib.request, "urlopen", open_index)
    assert smoke.select_package_index() == "https://mirrors.aliyun.com/pypi/simple"


def test_unreachable_indexes_abort_installation(monkeypatch):
    def fail(url, timeout):
        raise urllib.error.URLError("DNS unavailable")

    monkeypatch.setattr(smoke.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="No reachable package index"):
        smoke.select_package_index()


def test_missing_final_iteration_is_rejected(tmp_path):
    torch.save({"actor_model_state_dict": {"weight": torch.ones(1)}, "iter": 0}, tmp_path / "model_00000.pt")
    with pytest.raises(RuntimeError, match="before the requested final iteration"):
        smoke.verify_artifacts(tmp_path, 2)


def test_checkpoint_and_real_bundled_onnx_are_executable(tmp_path):
    pytest.importorskip("onnxruntime")
    torch.save({"actor_model_state_dict": {"weight": torch.ones(1)}, "iter": 1}, tmp_path / "model_00001.pt")
    model = ROOT / "src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/ppo_g1_29dof.onnx"
    shutil.copyfile(model, tmp_path / "model_00001.onnx")
    (tmp_path / "events.out.tfevents.test").write_bytes(b"test fixture")
    result = smoke.verify_artifacts(tmp_path, 2)
    assert result["onnx_action_shape"] == [1, 29]
    assert result["onnx_actions_finite"]
