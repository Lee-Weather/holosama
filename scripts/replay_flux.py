"""Replay a verified Holosoma checkpoint on Flux and publish an H.264 video.

The signed checkpoint URL is received only at runtime. Never print the URL,
its Base64 representation, or the complete task startScript.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import train_flux_smoke as bootstrap
from train_flux import repository_commit


def download_checkpoint(encoded_url: str, destination: Path, expected_sha256: str) -> None:
    """Download without leaking signed URLs through exception messages."""
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("A lowercase SHA-256 checkpoint digest is required")
    try:
        url = base64.urlsafe_b64decode(encoded_url + "=" * (-len(encoded_url) % 4)).decode("utf-8")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Invalid checkpoint location")
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:  # noqa: S310
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 128 * 1024 * 1024:
                    raise ValueError("Checkpoint exceeds 128 MiB")
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Checkpoint digest mismatch")
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint download or integrity check failed ({type(exc).__name__})") from None


def replay_configuration(saved, plan):
    """Build replay configuration without importing a simulator runtime."""
    import dataclasses  # noqa: PLC0415

    if plan.get("box_task"):
        from train_flux_box import check_task_config  # noqa: PLC0415

        check_task_config(saved.to_serializable_dict(), allow_resume=True)
        if saved.training.num_envs != 4096 or saved.algo.config.num_learning_iterations != 6000:
            raise RuntimeError("Box checkpoint does not match the verified continuation")
    elif saved.training.num_envs != 2048 or saved.algo.config.num_learning_iterations != 5000:
        raise RuntimeError("Checkpoint configuration does not match the formal training run")
    config = saved.get_eval_config()
    video = dataclasses.replace(
        config.logger.video,
        enabled=True,
        interval=1,
        width=1280,
        height=720,
        output_format="mp4",
        save_dir=plan["segments"],
        upload_to_wandb=False,
        use_recording_thread=False,
    )
    control_hz = config.simulator.config.sim.fps / config.simulator.config.sim.control_decimation_steps
    steps = round(plan["seconds"] * control_hz)
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            num_envs=1,
            headless=True,
            max_eval_steps=steps,
            project=plan["project"],
            name="g1_replay",
            export_onnx=False,
        ),
        logger=dataclasses.replace(config.logger, video=video, headless_recording=True),
        # Native evaluation otherwise uses a 100000-second timeout. A timeout
        # beyond the clip permits a continuous replay; physical falls still reset.
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config,
                sim=dataclasses.replace(
                    config.simulator.config.sim, max_episode_length_s=10.0 if plan.get("box_task") else 60.0
                ),
            ),
        ),
    )
    if plan.get("box_task"):
        config = dataclasses.replace(
            config,
            algo=dataclasses.replace(config.algo, config=dataclasses.replace(config.algo.config, load_optimizer=False)),
            observation=dataclasses.replace(
                config.observation,
                groups={
                    name: dataclasses.replace(group, enable_noise=False)
                    for name, group in config.observation.groups.items()
                },
            ),
        )
    return config


def run_worker(plan: dict) -> None:
    # Match eval_agent.py's import order: simulator modules are loaded by its
    # launcher, after IsaacSim starts. Use the native deterministic PPO evaluator.
    from holosoma.eval_agent import run_eval_with_tyro  # noqa: PLC0415
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config  # noqa: PLC0415

    checkpoint_cfg = CheckpointConfig(checkpoint=plan["checkpoint"])
    saved, wandb_path = load_saved_experiment_config(checkpoint_cfg)
    config = replay_configuration(saved, plan)
    control_hz = config.simulator.config.sim.fps / config.simulator.config.sim.control_decimation_steps
    steps = config.training.max_eval_steps
    print("FLUX_REPLAY_EVAL", json.dumps({"steps": steps, "control_hz": control_hz}), flush=True)
    sys.argv = ["eval_agent.py"]
    run_eval_with_tyro(config, checkpoint_cfg, saved, wandb_path)


def validate_video(path: Path, expected_seconds: int) -> dict:
    """Decode the entire file and reject empty, static, or truncated recordings."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = 0
        contrast = []
        motion = []
        previous = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            if frames % 25 == 0:
                sample = cv2.resize(frame, (160, 90)).astype(np.float32)
                contrast.append(float(sample.std()))
                if previous is not None:
                    motion.append(float(np.abs(sample - previous).mean()))
                previous = sample
    finally:
        capture.release()
    duration = frames / fps if fps > 0 else 0
    if (width, height) != (1280, 720) or not expected_seconds * 0.9 <= duration <= expected_seconds * 1.1:
        raise RuntimeError(f"Video dimensions/duration invalid: {width}x{height}, {duration:.3f}s")
    if not contrast or max(contrast) < 5 or not motion or max(motion) < 0.1:
        raise RuntimeError("Video contains empty or static frames")
    return {
        "frames": frames,
        "fps": fps,
        "duration_s": duration,
        "width": width,
        "height": height,
        "max_sample_contrast": max(contrast),
        "max_sample_frame_difference": max(motion),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def assemble_video(run_root: Path, expected_seconds: int) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # IsaacLab commonly provides imageio-ffmpeg's bundled executable.
        import imageio_ffmpeg  # noqa: PLC0415

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    segments = sorted((run_root / "segments").glob("*.mp4"), key=lambda path: path.stat().st_mtime_ns)
    if not segments:
        raise RuntimeError("Native replay produced no video segments")
    manifest = run_root / "segments.txt"
    # Paths are generated beneath our safe project name; no user-provided paths.
    manifest.write_text("".join(f"file '{path.resolve().as_posix()}'\n" for path in segments), encoding="utf-8")
    staging = run_root / "play_output.tmp"
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(staging),
        ],
        check=True,
        timeout=180,
    )
    result = validate_video(staging, expected_seconds)
    staging.replace(run_root / "play_output.mp4")
    return {"segments": len(segments), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=Path)
    parser.add_argument("--experiment", default="exp_1.1")
    parser.add_argument("--checkpoint_url_b64")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--install-timeout", type=int, default=900)
    parser.add_argument("--replay-timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.worker_plan:
        run_worker(json.loads(args.worker_plan.read_text(encoding="utf-8")))
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", args.experiment):
        parser.error("Invalid experiment name")
    if not 10 <= args.seconds <= 40 or not 1 <= args.install_timeout <= 1800 or not 1 <= args.replay_timeout <= 1800:
        parser.error("Clip must be 10..40s; installation/replay limits must be 1..1800s")
    if not args.checkpoint_url_b64 or not args.checkpoint_sha256:
        parser.error("Checkpoint download and integrity arguments are required")
    project = f"flux-{args.experiment}-replay-" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_root = (Path("logs") / project).resolve()
    (run_root / "segments").mkdir(parents=True)
    checkpoint = run_root / "model_04999.pt"
    download_checkpoint(args.checkpoint_url_b64, checkpoint, args.checkpoint_sha256)
    env = bootstrap.prepare_environment()
    bootstrap.install_training_package(env, args.install_timeout)
    bootstrap.check_gpu(env)
    # This checkpoint is from our own task and has a verified digest. The
    # existing native loader uses torch.load's default; explicitly scope trust
    # to this child interpreter instead of changing global PyTorch settings.
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    plan = {
        "commit": repository_commit(),
        "project": project,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "seconds": args.seconds,
        "segments": str(run_root / "segments"),
    }
    plan_path = run_root / "replay_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("FLUX_REPLAY_START", json.dumps(plan), flush=True)
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-plan", str(plan_path)],
        check=True,
        timeout=args.replay_timeout,
        env=env,
    )
    # Use the fresh interpreter after bootstrap installs OpenCV/NumPy.
    verify = (
        "import json,pathlib,sys; "
        f"sys.path.insert(0,{str(Path(__file__).resolve().parent)!r}); "
        "import replay_flux as r; "
        f"root=pathlib.Path({str(run_root)!r}); "
        f"result=r.assemble_video(root,{args.seconds}); "
        "(root/'replay_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); "
        "print('FLUX_REPLAY_SUCCESS',json.dumps(result),flush=True)"
    )
    subprocess.run([sys.executable, "-c", verify], check=True, timeout=240, env=env)
    print("FLUX_REPLAY_UPLOAD_WAIT 60s", flush=True)
    time.sleep(60)


if __name__ == "__main__":
    main()
