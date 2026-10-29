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
    monkeypatch.setattr(smoke, "_index_download_speed", lambda _index: 0.0)
    assert smoke.select_package_index() == "https://mirrors.aliyun.com/pypi/simple"


def test_reachable_slow_index_does_not_hide_faster_artifact_source(monkeypatch):
    monkeypatch.setattr(smoke, "_index_reachable", lambda _index: True)
    rates = [250_000, 8_000_000, 1_000_000]
    monkeypatch.setattr(smoke, "_index_download_speed", lambda _index: rates.pop(0))
    assert smoke.select_package_index() == "https://mirrors.aliyun.com/pypi/simple"


def test_failed_speed_probes_keep_reachable_index(monkeypatch):
    monkeypatch.setattr(smoke, "_index_reachable", lambda _index: True)
    monkeypatch.setattr(smoke, "_index_download_speed", lambda _index: 0.0)
    assert smoke.select_package_index() == "https://pypi.tuna.tsinghua.edu.cn/simple"


def test_speed_probe_is_bounded_and_does_not_download_entire_wheel(monkeypatch):
    class Wheel:
        def __init__(self):
            self.bytes_read = 0

        def read(self, size):
            self.bytes_read += size
            return b"x" * size

    wheel = Wheel()
    index_html = SimpleNamespace(
        read=lambda _size: (
            b'<a href="../../packages/'
            b'onnxruntime-1.20.1-cp311-cp311-manylinux_2_27_x86_64.whl">wheel</a>'
        )
    )
    responses = [index_html, wheel]
    requests = []

    def open_response(url, **_kwargs):
        requests.append(url)
        return nullcontext(responses.pop(0))

    monkeypatch.setattr(smoke.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(smoke.time, "monotonic", lambda: 0.0)
    assert smoke._index_download_speed("https://mirrors.aliyun.com/pypi/simple") > 0
    assert wheel.bytes_read == 1024 * 1024
    assert requests[1].full_url.startswith("https://mirrors.aliyun.com/pypi/packages/onnxruntime-")


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
