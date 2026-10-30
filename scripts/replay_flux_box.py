"""Evaluate the final resumed box policy, then record native IsaacLab video.

Uses a repository checkpoint with an explicit digest; no signed URL in commands.
Runs 32x4 native episodes for comparison, then one environment for a 30s video.
Native failures and clip resets remain active; video duration is not success.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import train_flux_smoke as bootstrap
from train_flux import repository_commit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--install-timeout", type=int, default=900)
    parser.add_argument("--evaluation-timeout", type=int, default=900)
    parser.add_argument("--replay-timeout", type=int, default=900)
    parser.add_argument("--video-only", action="store_true", help="Keep prior native evaluation and only record video")
    args = parser.parse_args()
    env = bootstrap.prepare_environment()
    checkpoint = ROOT / "czy/data/exp_2.2/model_13999.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
        raise RuntimeError("Final checkpoint digest mismatch")
    if not all(1 <= value <= 1800 for value in (args.install_timeout, args.evaluation_timeout, args.replay_timeout)):
        parser.error("Stage timeouts must be 1..1800 seconds")
    project = "flux-exp_2.2-native-replay-" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output = ROOT / "logs" / project
    (output / "segments").mkdir(parents=True)
    plan = {
        "commit": repository_commit(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "project": project,
        "result_dir": str(output),
        "num_envs": 32,
        "episodes_per_env": 4,
        "max_steps": 2500,
        "expected_iteration": 13999,
        "source_num_envs": 4096,
        "box_task": True,
        "seconds": 30,
        "segments": str(output / "segments"),
        "video_only": args.video_only,
    }
    path = output / "native_replay_plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("FLUX_BOX_NATIVE_REPLAY_PLAN", json.dumps(plan), flush=True)
    bootstrap.install_training_package(env, args.install_timeout)
    bootstrap.check_gpu(env)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["LOGURU_LEVEL"] = "INFO"
    stages = [("replay_flux.py", args.replay_timeout)]
    if not args.video_only:
        stages.insert(0, ("eval_flux_box.py", args.evaluation_timeout))
    for script, timeout in stages:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--worker-plan", str(path)],
            check=True,
            timeout=timeout,
            env=env,
        )
        if script == "eval_flux_box.py":
            result = json.loads((output / "native_eval_result.json").read_text(encoding="utf-8"))
            encoded = base64.b64encode(gzip.compress(json.dumps(result).encode())).decode()
            print("FLUX_BOX_NATIVE_RESULT_GZIP", json.dumps({"payload_b64": encoded}), flush=True)
    # Bootstrap modifies site-packages: decode/assemble in a fresh interpreter.
    code = (
        "import json,pathlib,sys; "
        f"sys.path.insert(0,{str(ROOT / 'scripts')!r}); import replay_flux; "
        f"root=pathlib.Path({str(output)!r}); result=replay_flux.assemble_video(root,30,'play_output_13999.mp4'); "
        "(root/'video_result.json').write_text(json.dumps(result,indent=2)); "
        "print('FLUX_BOX_NATIVE_VIDEO_SUCCESS',json.dumps(result),flush=True)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=240, env=env)
    print("FLUX_BOX_NATIVE_REPLAY_SUCCESS", flush=True)
    time.sleep(60)


if __name__ == "__main__":
    main()
