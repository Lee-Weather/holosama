"""Protect checkpoint integrity and reject unusable replay recordings."""

import base64
import hashlib
import importlib
import io
from pathlib import Path

import pytest


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("replay_flux")


def test_signed_location_is_not_exposed_on_download_failure(replay, monkeypatch, tmp_path):
    location = "https://example.com/model.pt?private=do-not-print"

    def fail(*_args, **_kwargs):
        raise OSError(location)

    monkeypatch.setattr(replay.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError) as caught:
        replay.download_checkpoint(
            base64.urlsafe_b64encode(location.encode()).decode(), tmp_path / "model.pt", "a" * 64
        )
    assert "example.com" not in str(caught.value)
    assert "do-not-print" not in str(caught.value)


def test_integrity_mismatch_removes_checkpoint(replay, monkeypatch, tmp_path):
    monkeypatch.setattr(replay.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(b"bad checkpoint"))
    destination = tmp_path / "model.pt"
    encoded = base64.urlsafe_b64encode(b"https://example.com/model.pt").decode()
    with pytest.raises(RuntimeError):
        replay.download_checkpoint(encoded, destination, hashlib.sha256(b"correct checkpoint").hexdigest())
    assert not destination.exists()


def test_local_file_location_is_rejected_before_open(replay, monkeypatch, tmp_path):
    monkeypatch.setattr(replay.urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("Invalid URL reached network"))
    with pytest.raises(RuntimeError):
        replay.download_checkpoint(
            base64.urlsafe_b64encode(b"file:///private").decode(), tmp_path / "model.pt", "a" * 64
        )


def test_publish_creates_indexed_mp4_instead_of_renaming_tmp(replay, monkeypatch, tmp_path):
    segments = tmp_path / "segments"
    segments.mkdir()
    (segments / "episode_1.mp4").write_bytes(b"segment")
    monkeypatch.setattr(replay.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(
        replay.subprocess, "run", lambda *_a, **_k: (tmp_path / "play_output.tmp").write_bytes(b"encoded")
    )
    monkeypatch.setattr(replay, "validate_video", lambda *_: {"bytes": 7})
    monkeypatch.setattr(Path, "replace", lambda *_: pytest.fail("Rename does not trigger Flux upload creation"))
    created = []
    original_open = replay.shutil.copyfile

    def publish(src, dst):
        assert not dst.exists() and dst.name == "play_output_13999.mp4"
        created.append(dst)
        return original_open(src, dst)

    monkeypatch.setattr(replay.shutil, "copyfile", publish)
    result = replay.assemble_video(tmp_path, 30, "play_output_13999.mp4")
    assert len(created) == 1 and created[0].read_bytes() == b"encoded"
    assert result["filename"] == "play_output_13999.mp4" and not (tmp_path / "play_output.tmp").exists()


@pytest.mark.parametrize("kind", ["valid", "black", "static", "truncated"])
def test_video_acceptance_decodes_content(replay, tmp_path, kind):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path / "test.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    assert writer.isOpened()
    for index in range(20 if kind == "truncated" else 100):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        if kind != "black":
            x = 100 if kind == "static" else index * 10
            cv2.rectangle(frame, (x, 100), (x + 150, 600), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    if kind == "valid":
        result = replay.validate_video(path, 10)
        assert result["frames"] == 100 and result["duration_s"] == 10
    else:
        with pytest.raises(RuntimeError):
            replay.validate_video(path, 10)
