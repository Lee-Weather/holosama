"""Train G1 whole-body tracking on Flux.

Single-motion mode keeps the original sub10 reference. Multi-motion mode loads
every .npz under --motion-dir (package-relative, e.g.
``holosoma/data/motions/g1_29dof/whole_body_tracking/carry_segments``).
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


def training_command(
    num_envs: int,
    iterations: int,
    save_interval: int,
    project: str,
    *,
    motion: str,
    motion_dir: str,
    name: str,
) -> list[str]:
    if not 1 <= num_envs <= 4096 or not 1 <= iterations <= 30000:
        raise ValueError("WBT training requires 1..4096 environments and 1..30000 iterations")
    if not 1 <= save_interval <= iterations:
        raise ValueError("Invalid save interval")
    if bool(motion) == bool(motion_dir):
        raise ValueError("Set exactly one of motion_file and motion_dir")
    motion_override = (
        f"--command.setup_terms.motion_command.params.motion_config.motion_dir={motion_dir}"
        if motion_dir
        else f"--command.setup_terms.motion_command.params.motion_config.motion_file={motion}"
    )
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
        f"--training.name={name}",
        f"--algo.config.num-learning-iterations={iterations}",
        f"--algo.config.save-interval={save_interval}",
        "--logger.video.enabled=False",
        motion_override,
    ]


def _motion_root(relative: str) -> Path:
    return bootstrap.REPO_ROOT / "src/holosoma" / relative


def preflight(motion: str, motion_dir: str) -> dict:
    """Verify the reference file or directory and fingerprint its contents."""
    if motion_dir:
        root = _motion_root(motion_dir)
        files = sorted(root.glob("*.npz"))
        if not files:
            details: dict = {}
            try:
                details["root_exists"] = root.exists()
                details["parent_exists"] = root.parent.exists()
                details["parent"] = str(root.parent)
                details["parent_children"] = (
                    sorted(p.name for p in root.parent.iterdir())[:50] if root.parent.exists() else None
                )
                details["root_children"] = (
                    sorted(p.name for p in root.iterdir())[:20] if root.exists() else None
                )
                for rel in ("", "src", "src/holosoma", "src/holosoma/holosoma",
                            "src/holosoma/holosoma/data", "src/holosoma/holosoma/data/motions",
                            "src/holosoma/holosoma/data/motions/g1_29dof"):
                    probe = bootstrap.REPO_ROOT / rel
                    details[f"probe:{rel or 'repo'}"] = (
                        sorted(p.name for p in probe.iterdir())[:30] if probe.exists() else None
                    )
                import os as _os
                for rel in ("src/holosoma/holosoma/data",
                            "src/holosoma/holosoma/data/motions",
                            "src/holosoma/holosoma/data/motions/g1_29dof",
                            "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"):
                    probe = bootstrap.REPO_ROOT / rel
                    details[f"link:{rel}"] = {
                        "exists": probe.exists(),
                        "is_symlink": probe.is_symlink(),
                        "readlink": _os.readlink(probe) if probe.is_symlink() else None,
                    }
            except Exception as exc:  # diagnostics must not hide the original failure
                details["scan_error"] = repr(exc)
            raise FileNotFoundError(f"No .npz files under {root}: {details}")
        digest = hashlib.sha256()
        total_bytes = 0
        for path in files:
            data = path.read_bytes()
            total_bytes += len(data)
            digest.update(path.name.encode())
            digest.update(hashlib.sha256(data).digest())
        summary = {
            "mode": "motion_dir",
            "motion_dir": motion_dir,
            "files": len(files),
            "bytes": total_bytes,
            "manifest_sha256": digest.hexdigest(),
            "first": files[0].name,
            "last": files[-1].name,
        }
        print("FLUX_WBT_PREFLIGHT", json.dumps(summary), flush=True)
        return summary

    motion_path = _motion_root(motion)
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    summary = {
        "mode": "motion_file",
        "motion": motion,
        "sha256": hashlib.sha256(motion_path.read_bytes()).hexdigest(),
        "files": 1,
        "bytes": motion_path.stat().st_size,
    }
    print("FLUX_WBT_PREFLIGHT", json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="exp_3")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--install-timeout", type=int, default=900)
    parser.add_argument("--training-timeout", type=int, default=86400)
    parser.add_argument("--motion", default="", help="Package-relative single .npz (default: sub10 reference)")
    parser.add_argument("--motion-dir", default="", help="Package-relative directory or glob of .npz files")
    parser.add_argument("--name", default="g1_wbt", help="Training run name")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", args.experiment):
        parser.error("Invalid experiment name")
    if not 1 <= args.install_timeout <= 1800 or not 1 <= args.training_timeout <= 86400:
        parser.error("Install limit 1..1800s; training limit 1..86400s")
    if not args.motion and not args.motion_dir:
        args.motion = MOTION
    if bool(args.motion) == bool(args.motion_dir):
        parser.error("Pass exactly one of --motion (single file) or --motion-dir (multi file)")

    reference = preflight(args.motion, args.motion_dir)

    project = f"flux-{args.experiment}-wbt-" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    command = training_command(
        args.num_envs,
        args.iterations,
        args.save_interval,
        project,
        motion=args.motion,
        motion_dir=args.motion_dir,
        name=args.name,
    )

    env = bootstrap.prepare_environment()
    root = (Path("logs") / project).resolve()
    root.mkdir(parents=True, exist_ok=True)

    plan = {
        "commit": repository_commit(),
        "preset": PRESET,
        "reference": reference,
        "command": command,
        **vars(args),
    }
    (root / "launch_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("FLUX_WBT_PLAN", json.dumps(plan), flush=True)

    bootstrap.install_training_package(env, args.install_timeout)
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
