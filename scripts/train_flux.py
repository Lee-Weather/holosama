"""Run G1 PPO training on Flux using the dependency bootstrap validated by exp0.

Launch: gm-run holosama/scripts/train_flux.py --experiment exp_1
Defaults: 2048 environments, 5000 iterations, save every 100 iterations,
and a six-hour training time limit. No automatic retries or account switching.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import train_flux_smoke as bootstrap


def repository_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=bootstrap.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def training_command(num_envs: int, iterations: int, save_interval: int, project: str) -> list[str]:
    if not 1 <= num_envs <= 4096 or not 1 <= iterations <= 25000:
        raise ValueError("Training requires 1..4096 environments and 1..25000 iterations")
    if not 1 <= save_interval <= iterations:
        raise ValueError("Save interval must be positive and no larger than the iteration count")
    return [
        sys.executable,
        str(bootstrap.REPO_ROOT / "src/holosoma/holosoma/train_agent.py"),
        "exp:g1-29dof",
        "simulator:isaacsim",
        "logger:disabled",
        "terrain:terrain-locomotion-plane",
        f"--training.num-envs={num_envs}",
        "--training.headless=True",
        "--training.seed=1",
        f"--training.project={project}",
        "--training.name=g1_ppo",
        f"--algo.config.num-learning-iterations={iterations}",
        f"--algo.config.save-interval={save_interval}",
        "--logger.video.enabled=False",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="exp_1")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--install-timeout", type=int, default=900)
    parser.add_argument("--training-timeout", type=int, default=21600)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment):
        parser.error("Experiment name must contain only letters, digits, underscores, or hyphens")
    if not 1 <= args.install_timeout <= 1800 or not 1 <= args.training_timeout <= 86400:
        parser.error("Installation limit must be 1..1800s; training limit must be 1..86400s")
    project = f"flux-{args.experiment}-" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    command = training_command(args.num_envs, args.iterations, args.save_interval, project)
    env = bootstrap.prepare_environment()
    run_root = Path("logs") / project
    run_root.mkdir(parents=True, exist_ok=True)
    plan = {"commit": repository_commit(), "command": command, **vars(args)}
    (run_root / "launch_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("FLUX_TRAIN_PLAN", json.dumps(plan), flush=True)
    print("FLUX_TRAIN_BOOTSTRAP", json.dumps(bootstrap.package_versions()), flush=True)
    bootstrap.install_training_package(env, args.install_timeout)
    bootstrap.check_gpu(env)
    print("FLUX_TRAIN_START", project, flush=True)
    subprocess.run(command, check=True, timeout=args.training_timeout, env=env)
    # Run acceptance in a new interpreter, as the SDK wrapper may hold old imports.
    verify = (
        "import importlib.util,json,pathlib; "
        f"s=importlib.util.spec_from_file_location('flux_bootstrap',{str(Path(bootstrap.__file__).resolve())!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"root=pathlib.Path({str(run_root)!r}); "
        f"r=m.verify_artifacts(root,{args.iterations}); "
        f"r.update(success=True,experiment={args.experiment!r},versions=m.package_versions()); "
        "(root/'training_result.json').write_text(json.dumps(r,indent=2),encoding='utf-8'); "
        "print('FLUX_TRAIN_SUCCESS',json.dumps(r),flush=True)"
    )
    subprocess.run([sys.executable, "-c", verify], check=True, timeout=120, env=env)
    print("FLUX_TRAIN_UPLOAD_WAIT 60s", flush=True)
    time.sleep(60)


if __name__ == "__main__":
    main()
