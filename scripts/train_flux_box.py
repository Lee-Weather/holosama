"""Train G1 to track the bundled box-carry motion with a dynamic box on Flux.

A 128-env/two-iteration smoke run must pass before the 1024-env/8000-iteration
first training stage starts. Both stages use g1-29dof-wbt-w-object from scratch.
Installation, smoke training, and formal training have separate time limits.
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

PRESET = "exp:g1-29dof-wbt-w-object"
MOTION = "holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
MOTION_SHA256 = "76077a5035726aabe25b084de5414f3113646c43ac1bfa6ccc0742e46b1a4cea"
ROBOT_URDF = "g1/main_mesh_collision_halfspherehand.urdf"
BOX_URDF = "holosoma/data/scene_objects/boxes/large_box.urdf"


def command(num_envs: int, iterations: int, save_interval: int, project: str) -> list[str]:
    if not 1 <= num_envs <= 2048 or not 2 <= iterations <= 30000:
        raise ValueError("Box training requires 1..2048 environments and 2..30000 iterations")
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
        "--training.name=g1_box_carry_ppo",
        f"--algo.config.num-learning-iterations={iterations}",
        f"--algo.config.save-interval={save_interval}",
        "--logger.video.enabled=False",
    ]


def check_task_config(cfg: dict) -> None:
    """Fail closed if this is robot-only WBT or the old locomotion task."""
    motion = cfg["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["motion_file"]
    if cfg["env_class"] != "holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager" or motion != MOTION:
        raise ValueError("Expected the object-carry WBT environment and object motion")
    if cfg["scene"]["rigid_objects"].get("object", {}).get("urdf_file") != BOX_URDF:
        raise ValueError("Dynamic box scene is missing")
    if cfg["scene"]["rigid_objects"]["object"].get("fixed", False):
        raise ValueError("The box must be a free dynamic body")
    if cfg["robot"]["asset"]["urdf_file"] != ROBOT_URDF:
        raise ValueError("Object interaction requires the configured hand collision model")
    rewards = cfg["reward"]["terms"]
    for key in ("object_global_ref_position_error_exp", "object_global_ref_orientation_error_exp"):
        if key not in rewards or rewards[key]["weight"] <= 0:
            raise ValueError("Object position/orientation rewards must be active")
    if "obj_pos_b" not in cfg["observation"]["groups"]["critic_obs"]["terms"]:
        raise ValueError("Critic object-state observations are missing")
    if cfg["training"]["checkpoint"] is not None:
        raise ValueError("This experiment starts from scratch; no locomotion checkpoint is allowed")


def preflight(run_root: Path) -> None:
    import numpy as np  # noqa: PLC0415
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object  # noqa: PLC0415

    cfg = g1_29dof_wbt_w_object.to_serializable_dict()
    check_task_config(cfg)
    motion = bootstrap.REPO_ROOT / "src/holosoma" / MOTION
    if hashlib.sha256(motion.read_bytes()).hexdigest() != MOTION_SHA256:
        raise ValueError("Unexpected box motion content")
    with np.load(motion, allow_pickle=False) as data:
        required = {
            "fps",
            "joint_names",
            "body_names",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "object_pos_w",
            "object_quat_w",
            "object_lin_vel_w",
        }
        if not required <= set(data.files):
            raise ValueError("Motion data is missing robot/object trajectories")
        for key in required - {"joint_names", "body_names"}:
            if not np.isfinite(data[key]).all():
                raise ValueError(f"Motion has nonfinite {key}")
        joint_names = data["joint_names"].tolist()
        if set(joint_names) != set(cfg["robot"]["dof_names"]):
            raise ValueError("Reference motion does not match the robot joint names")
        frames = len(data["joint_pos"])
        if data["joint_pos"].shape != (frames, 36) or data["object_pos_w"].shape != (frames, 3):
            raise ValueError("Unexpected reference motion dimensions")
        fps = float(data["fps"].item())
        result = {
            "preset": PRESET,
            "motion_sha256": MOTION_SHA256,
            "frames": frames,
            "fps": fps,
            "sample_duration_s": frames / fps,
            "object_height_range_m": [float(data["object_pos_w"][:, 2].min()), float(data["object_pos_w"][:, 2].max())],
            "robot_urdf": ROBOT_URDF,
            "box_urdf": BOX_URDF,
            "object_reward_terms": [k for k in cfg["reward"]["terms"] if k.startswith("object_")],
        }
    asset_root = bootstrap.REPO_ROOT / "src/holosoma/holosoma/data/robots"
    if not (asset_root / ROBOT_URDF).is_file() or not (bootstrap.REPO_ROOT / "src/holosoma" / BOX_URDF).is_file():
        raise ValueError("Robot or box asset is missing")
    (run_root / "box_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FLUX_BOX_PREFLIGHT", json.dumps(result), flush=True)


def verify(directory: Path, iterations: int, num_envs: int) -> dict:
    import torch  # noqa: PLC0415

    result = bootstrap.verify_artifacts(directory, iterations)
    saved = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    cfg = saved["experiment_config"]
    check_task_config(cfg)
    if cfg["training"]["num_envs"] != num_envs:
        raise ValueError("Saved environment count differs from the requested box training stage")
    for key in ("actor_model_state_dict", "critic_model_state_dict", "actor_obs_normalizer_state_dict"):
        if not saved.get(key):
            raise ValueError(f"Missing {key}")
        for value in saved[key].values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {key}")
    return {**result, "preset": PRESET, "num_envs": num_envs, "object_task_verified": True}


def fresh(code: str, env: dict, timeout: int) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            f"sys.path.insert(0,{str(Path(__file__).resolve().parent)!r}); "
            "from pathlib import Path; import json; import train_flux_box as box; " + code,
        ],
        check=True,
        timeout=timeout,
        env=env,
    )


def verify_fresh(directory: Path, iterations: int, num_envs: int, env: dict, marker: str) -> None:
    fresh(
        f"root=Path({str(directory)!r}); r=box.verify(root,{iterations},{num_envs}); "
        "(root/'box_result.json').write_text(json.dumps(r,indent=2),encoding='utf-8'); "
        f"print({marker!r},json.dumps(r),flush=True)",
        env,
        120,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="exp_2")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--install-timeout", type=int, default=900)
    parser.add_argument("--smoke-timeout", type=int, default=900)
    parser.add_argument("--training-timeout", type=int, default=21600)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", args.experiment):
        parser.error("Invalid experiment name")
    if (
        not 1 <= args.install_timeout <= 1800
        or not 1 <= args.smoke_timeout <= 1800
        or not 1 <= args.training_timeout <= 43200
    ):
        parser.error("Installation/smoke limits must be 1..1800s; training limit must be 1..43200s")
    project = f"flux-{args.experiment}-box-" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    full_command = command(args.num_envs, args.iterations, args.save_interval, project)
    smoke_project = project + "-smoke"
    smoke_command = command(128, 2, 1, smoke_project)
    # Bootstrap changes cwd to the clone root. Establish it before creating
    # paths, then keep the project root absolute across all child interpreters.
    env = bootstrap.prepare_environment()
    root = (Path("logs") / project).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "commit": repository_commit(),
        "preset": PRESET,
        "motion": MOTION,
        "motion_sha256": MOTION_SHA256,
        "smoke_command": smoke_command,
        "training_command": full_command,
        "resume": False,
        **vars(args),
    }
    (root / "launch_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("FLUX_BOX_PLAN", json.dumps(plan), flush=True)
    bootstrap.install_training_package(env, args.install_timeout)
    fresh(f"box.preflight(Path({str(root)!r}))", env, 60)
    bootstrap.check_gpu(env)
    print("FLUX_BOX_SMOKE_START", smoke_project, flush=True)
    subprocess.run(smoke_command, check=True, timeout=args.smoke_timeout, env=env)
    verify_fresh(Path("logs") / smoke_project, 2, 128, env, "FLUX_BOX_SMOKE_SUCCESS")
    print("FLUX_BOX_TRAIN_START", project, flush=True)
    subprocess.run(full_command, check=True, timeout=args.training_timeout, env=env)
    verify_fresh(root, args.iterations, args.num_envs, env, "FLUX_BOX_TRAIN_SUCCESS")
    print("FLUX_BOX_UPLOAD_WAIT 60s", flush=True)
    time.sleep(60)


if __name__ == "__main__":
    main()
