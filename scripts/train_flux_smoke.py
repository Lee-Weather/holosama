"""Run a bounded G1 PPO smoke test in Flux's preinstalled IsaacLab environment.

Launch with ``gm-run holosama/scripts/train_flux_smoke.py``. This entry point
installs the training package into gm-run's interpreter without rebuilding the
simulator, runs two PPO iterations, and checks the exported policy on CPU.
"""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def training_command(num_envs: int, iterations: int, project: str) -> list[str]:
    if not 1 <= num_envs <= 128 or not 2 <= iterations <= 10:
        raise ValueError("Smoke tests require 1..128 environments and 2..10 iterations")
    return [
        sys.executable,
        str(REPO_ROOT / "src/holosoma/holosoma/train_agent.py"),
        "exp:g1-29dof",
        "simulator:isaacsim",
        "logger:disabled",
        "terrain:terrain-locomotion-plane",
        f"--training.num-envs={num_envs}",
        "--training.headless=True",
        "--training.seed=1",
        f"--training.project={project}",
        "--training.name=g1_ppo_smoke",
        f"--algo.config.num-learning-iterations={iterations}",
        "--algo.config.save-interval=1",
        "--logger.video.enabled=False",
    ]


def verify_artifacts(directory: Path, iterations: int) -> dict:
    """Load our checkpoint and execute the exported ONNX graph, not just stat files."""
    # gm-run must be able to load this bootstrap before these dependencies exist.
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    import torch  # noqa: PLC0415

    checkpoints = sorted(directory.rglob("model_*.pt"))
    if not checkpoints:
        raise RuntimeError(f"No training checkpoint found in {directory}")
    checkpoint = checkpoints[-1]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or not saved.get("actor_model_state_dict"):
        raise RuntimeError("Checkpoint does not contain trained actor weights")
    if int(saved.get("iter", -1)) < iterations - 1:
        raise RuntimeError("Checkpoint was saved before the requested final iteration")
    onnx_path = checkpoint.with_suffix(".onnx")
    if not onnx_path.is_file():
        raise RuntimeError("Final checkpoint has no corresponding ONNX export")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {}
    for value in session.get_inputs():
        if value.type != "tensor(float)":
            raise RuntimeError(f"Unexpected PPO ONNX input type: {value.type}")
        shape = [n if isinstance(n, int) and n > 0 else 1 for n in value.shape]
        feeds[value.name] = np.zeros(shape, dtype=np.float32)
    outputs = session.run(None, feeds)
    if not outputs or outputs[0].shape != (1, 29) or not np.isfinite(outputs[0]).all():
        raise RuntimeError("Exported policy did not produce 29 finite G1 joint actions")
    event_files = list(directory.rglob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("Training did not produce a TensorBoard event file")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "final_iteration": int(saved["iter"]),
        "onnx": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_action_shape": list(outputs[0].shape),
        "onnx_actions_finite": True,
        "tensorboard_files": len(event_files),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed as a distribution"


def package_versions() -> dict[str, str]:
    return {
        name: _package_version(name)
        for name in ("isaacsim", "isaaclab", "torch", "numpy", "pydantic", "tyro", "tensordict", "warp-lang")
    }


def _index_reachable(index: str) -> bool:
    try:
        # Called only with the fixed HTTPS package-index allowlist below.
        with urllib.request.urlopen(index + "/uv-build/", timeout=8) as response:  # noqa: S310
            return response.status == 200
    except OSError as error:
        print("FLUX_SMOKE_INDEX_UNAVAILABLE", urllib.parse.urlparse(index).hostname, type(error).__name__, flush=True)
        return False


def _index_download_speed(index: str) -> float:
    """Sample up to 1 MiB of a known Linux wheel; do not install the probe wheel."""
    try:
        with urllib.request.urlopen(index + "/onnxruntime/", timeout=6) as response:  # noqa: S310
            links = re.findall(r"href=[\"']([^\"']+)", response.read(2 * 1024 * 1024).decode("utf-8"))
        for link in links:
            url = urllib.parse.urljoin(index + "/onnxruntime/", html.unescape(link))
            parsed = urllib.parse.urlparse(url)
            filename = parsed.path.rsplit("/", 1)[-1]
            if not (
                filename.startswith("onnxruntime-1.20.1-cp311-cp311-manylinux") and filename.endswith("x86_64.whl")
            ):
                continue
            if parsed.scheme != "https" or parsed.hostname not in {
                "files.pythonhosted.org",
                "pypi.tuna.tsinghua.edu.cn",
                "mirrors.tuna.tsinghua.edu.cn",
                "mirrors.aliyun.com",
            }:
                continue
            request = urllib.request.Request(url, headers={"Range": "bytes=0-1048575"})  # noqa: S310
            started = time.monotonic()
            received = 0
            # URL is HTTPS and its host is explicitly checked above.
            with urllib.request.urlopen(request, timeout=6) as response:  # noqa: S310
                while received < 1024 * 1024 and time.monotonic() - started < 6:
                    chunk = response.read(min(64 * 1024, 1024 * 1024 - received))
                    if not chunk:
                        break
                    received += len(chunk)
            speed = received / max(time.monotonic() - started, 0.001)
            print("FLUX_BOOTSTRAP_INDEX_SPEED", parsed.hostname, round(speed / 1024), "KiB/s", flush=True)
            return speed
    except OSError as error:
        print(
            "FLUX_BOOTSTRAP_SPEED_UNAVAILABLE", urllib.parse.urlparse(index).hostname, type(error).__name__, flush=True
        )
    return 0.0


def select_package_index() -> str:
    """Select a reachable index by measured artifact throughput, not HTTP 200 alone."""
    indexes = (
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "https://mirrors.aliyun.com/pypi/simple",
        "https://pypi.org/simple",
    )
    candidates = [(index, _index_download_speed(index)) for index in indexes if _index_reachable(index)]
    if candidates:
        # Stable ties preserve the domestic-first ordering; failed speed probes
        # do not discard a reachable index when all throughput probes fail.
        selected, _ = max(candidates, key=lambda item: item[1])
        print("FLUX_SMOKE_PIP_INDEX", urllib.parse.urlparse(selected).hostname, flush=True)
        return selected
    raise RuntimeError("No reachable package index; aborting before dependency installation")


def prepare_environment() -> dict[str, str]:
    """Keep training and SDK in the image interpreter while refreshing child imports."""
    os.chdir(REPO_ROOT)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # New subprocesses see newly installed packages even if gm-run imported older
    # versions into the wrapper process. The outer SDK can keep watching outputs.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src/holosoma") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def install_training_package(env: dict[str, str], timeout: int) -> None:
    index = select_package_index()
    env["PIP_INDEX_URL"] = index
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_RETRIES"] = "1"
    env["PIP_DEFAULT_TIMEOUT"] = "30"
    env.pop("PIP_EXTRA_INDEX_URL", None)
    env.pop("PIP_NO_INDEX", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--index-url",
            index,
            "--prefer-binary",
            "--timeout",
            "30",
            "--retries",
            "1",
            "-c",
            "scripts/flux_smoke_constraints.txt",
            "-e",
            "./src/holosoma",
        ],
        check=True,
        timeout=timeout,
        env=env,
    )


def check_gpu(env: dict[str, str]) -> None:
    probe = (
        "import json,torch; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "print('FLUX_SMOKE_GPU', json.dumps({'torch':torch.__version__,"
        "'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory}))"
    )
    subprocess.run([sys.executable, "-c", probe], check=True, timeout=60, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--install-timeout", type=int, default=900)
    parser.add_argument("--training-timeout", type=int, default=600)
    args = parser.parse_args()
    run_name = "flux-smoke-" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    command = training_command(args.num_envs, args.iterations, run_name)
    env = prepare_environment()
    print("FLUX_SMOKE_BOOTSTRAP", json.dumps({"python": sys.executable, "versions": package_versions()}), flush=True)
    if not args.skip_install:
        install_training_package(env, args.install_timeout)
    check_gpu(env)
    print("FLUX_SMOKE_TRAIN", json.dumps({"envs": args.num_envs, "iterations": args.iterations}), flush=True)
    subprocess.run(command, check=True, timeout=args.training_timeout, env=env)
    # Verify in a fresh process too, so dependency changes during bootstrap cannot
    # leave stale numpy/torch modules in the interpreter used for verification.
    verify = (
        "import importlib.util,json,pathlib; "
        f"s=importlib.util.spec_from_file_location('flux_smoke',{str(Path(__file__).resolve())!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"root=pathlib.Path('logs')/{run_name!r}; "
        f"r=m.verify_artifacts(root,{args.iterations}); "
        "r.update(success=True,versions=m.package_versions()); "
        "(root/'smoke_result.json').write_text(json.dumps(r,indent=2),encoding='utf-8'); "
        "print('FLUX_SMOKE_SUCCESS',json.dumps(r),flush=True)"
    )
    subprocess.run([sys.executable, "-c", verify], check=True, timeout=120, env=env)
    # Give the platform's external file scanner a bounded opportunity to upload.
    print("FLUX_SMOKE_UPLOAD_WAIT 60s", flush=True)
    time.sleep(60)


if __name__ == "__main__":
    main()
