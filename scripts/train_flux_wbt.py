"""Train G1 to track sub10 body motion (no box) on Flux.

Uses exp:g1-29dof-wbt with the sub10 reference motion, 4096 environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import train_flux_smoke as bootstrap
from train_flux import repository_commit

PRESET = "exp:g1-29dof-wbt"
MOTION = "holosoma/data/motions/g1_29dof/whole_body_tracking/sub10_largebox_049_mj.npz"


def training_command(num_envs: int, iterations: int, save_interval: int, project: str) -> list[str]:
    if not 1 <= num_envs <= 4096 or not 1 <= iterations <= 30000:
        raise ValueError("WBT training requires 1..4096 environments and 1..30000 iterations")
    if not 1 <= save_interval <= iterations:
        raise ValueError("Invalid save interval")
    return [
        sys.executable,
        str(bootstrap.REPO_ROOT / "src/holosoma/holosoma/train_agent.py"),
        PRESET,
        "simulator:isaacsim",
        "logger:disabled",
        f"--training.num-envs={num_envs}",
        "--training.headless=True",
        "--training.seed=1",
        f"--training.project={project}",
        "--training.name=g1_wbt_sub10",
        f"--algo.config.num-learning-iterations={iterations}",
        f"--algo.config.save-interval={save_interval}",
        "--logger.video.enabled=False",
        f'--command.setup_terms.motion_command.params.motion_config.motion_file="{MOTION}"',
    ]


def preflight() -> None:
    """Verify the motion file exists and has expected content."""
    motion_path = bootstrap.REPO_ROOT / "src" / MOTION
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    with open(motion_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"FLUX_WBT_PREFLIGHT motion={MOTION} sha256={sha256}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="exp_3")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--install-timeout", type=int, default=900)
    parser.add_argument("--training-timeout", type=int, default=86400)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", args.experiment):
        parser.error("Invalid experiment name")
    if not 1 <= args.install_timeout <= 1800 or not 1 <= args.training_timeout <= 86400:
        parser.error("Install limit 1..1800s; training limit 1..86400s")

    project = f"flux-{args.experiment}-wbt-" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    command = training_command(args.num_envs, args.iterations, args.save_interval, project)

    env = bootstrap.prepare_environment()
    root = (Path("logs") / project).resolve()
    root.mkdir(parents=True, exist_ok=True)

    plan = {
        "commit": repository_commit(),
        "preset": PRESET,
        "motion": MOTION,
        "command": command,
        **vars(args),
    }
    (root / "launch_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("FLUX_WBT_PLAN", json.dumps(plan), flush=True)

    bootstrap.install_training_package(env, args.install_timeout)
    preflight()
    bootstrap.check_gpu(env)
    print("FLUX_WBT_START", project, flush=True)
    subprocess.run(command, check=True, timeout=args.training_timeout, env=env)

    verify = (
        "import importlib.util,json,pathlib; "
        f"s=importlib.util.spec_from_file_location('flux_bootstrap',{str(Path(bootstrap.__file__).resolve())!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"root=pathlib.Path({str(root)!r}); "
        f"r=m.verify_artifacts(root,{args.iterations}); "
        f"r.update(success=True,experiment={args.experiment!r},versions=m.package_versions()); "
        "(root/'training_result.json').write_text(json.dumps(r,indent=2),encoding='utf-8'); "
        "print('FLUX_WBT_SUCCESS',json.dumps(r),flush=True)"
    )
    subprocess.run([sys.executable, "-c", verify], check=True, timeout=120, env=env)
    print("FLUX_WBT_UPLOAD_WAIT 60s", flush=True)
    time.sleep(60)


if __name__ == "__main__":
    main()
