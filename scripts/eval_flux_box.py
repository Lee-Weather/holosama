"""Evaluate the existing box policy in native IsaacLab without training or RTX video.

The original checkpoint is checked by digest and loaded from the checkout.
Termination components are observed before native episode resets; no policy,
termination threshold, body dynamics or contact conditions are changed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import train_flux_smoke as bootstrap
from train_flux import repository_commit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "e2d120660bbc199d0dc03d0bce17a4989ef6117877c2f97f9da72d2c4dbc2c97"


def evaluation_config(saved, plan):
    """Keep saved physics/DR and the 10-second timeout; disable observation noise/video."""
    config = saved.get_eval_config()
    return dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            num_envs=plan["num_envs"],
            headless=True,
            project=plan["project"],
            name="g1_box_native_eval",
            export_onnx=False,
        ),
        algo=dataclasses.replace(config.algo, config=dataclasses.replace(config.algo.config, load_optimizer=False)),
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config,
                sim=dataclasses.replace(config.simulator.config.sim, max_episode_length_s=10.0),
            ),
        ),
        logger=dataclasses.replace(
            config.logger,
            video=dataclasses.replace(config.logger.video, enabled=False),
            headless_recording=False,
        ),
        observation=dataclasses.replace(
            config.observation,
            groups={
                name: dataclasses.replace(group, enable_noise=False)
                for name, group in config.observation.groups.items()
            },
        ),
    )


def summarize(episodes, expected):
    if len(episodes) != expected:
        raise RuntimeError(f"Only {len(episodes)}/{expected} native episodes completed within the step limit")
    counts = Counter(reason for episode in episodes for reason in episode["termination_reasons"])
    successes = sum(episode["first_clip_completed"] for episode in episodes)
    return {
        "completed_episodes": len(episodes),
        "first_clip_completed_episodes": successes,
        "first_clip_success_rate": successes / len(episodes),
        "mean_episode_seconds": statistics.mean(row["duration_s"] for row in episodes),
        "min_episode_seconds": min(row["duration_s"] for row in episodes),
        "max_episode_seconds": max(row["duration_s"] for row in episodes),
        "termination_counts_overlapping": dict(counts),
    }


class NativeObserver:
    """Read-only instrumentation of the native pre-reset termination check."""

    def __init__(self, env, episodes_per_env, quaternion_error):
        import torch  # noqa: PLC0415

        self.torch = torch
        self.quaternion_error = quaternion_error
        self.env = env
        self.goal = episodes_per_env
        self.counts = [0] * env.num_envs
        self.clip_success = [False] * env.num_envs
        self.episodes = []
        self.active = False
        self.steps = 0
        self.samples = 0
        self.position_squared_sum = 0.0
        self.object_angle_sum = 0.0
        self.max_box_height = float("-inf")
        self.min_root_height = float("inf")
        self.original_check = env.termination_manager.check
        env.termination_manager.check = self.check

    def check(self):
        terminated, timeout = self.original_check()
        if not self.active:
            return terminated, timeout
        torch, env = self.torch, self.env
        motion = env.command_manager.get_state("motion_command")
        bad = env.termination_manager._term_instances["bad_tracking"]
        components = {
            name: getattr(bad, name)(motion)
            for name in (
                "bad_ref_pos",
                "bad_ref_ori",
                "bad_motion_body_pos",
                "bad_object_pos",
                "bad_object_ori",
            )
        }
        components["timeout"] = timeout
        # This task uses BadTrackingZOnly: preserve its exact per-body Z test.
        ids = bad.bad_motion_body_pos_body_indexes
        body_errors = torch.abs(motion.body_pos_relative_w[:, ids, 2] - motion.robot_body_pos_w[:, ids, 2])
        for index, name in enumerate(bad.bad_motion_body_pos_body_names):
            components["body_height:" + name] = body_errors[:, index] > bad.bad_motion_body_pos_threshold
        union = torch.zeros_like(terminated)
        for name in ("bad_ref_pos", "bad_ref_ori", "bad_motion_body_pos", "bad_object_pos", "bad_object_ori"):
            union |= components[name]
        if not torch.equal(union, terminated):
            raise RuntimeError("Diagnostic components do not match native termination flags")

        active = torch.tensor([n < self.goal for n in self.counts], dtype=torch.bool, device=env.device)
        positions = motion.simulator_object_pos_w
        errors = torch.norm(positions - motion.object_pos_w, dim=1)
        angles = self.quaternion_error(motion.object_quat_w, motion.simulator_object_quat_w)
        roots = motion.robot_root_pos_w
        if not torch.isfinite(positions).all() or not torch.isfinite(roots).all() or not torch.isfinite(errors).all():
            raise RuntimeError("Non-finite native simulator state")
        self.steps += 1
        self.samples += int(active.sum().item())
        if active.any():
            self.position_squared_sum += float(errors[active].square().sum().item())
            self.object_angle_sum += float(angles[active].sum().item())
            heights = positions[:, 2] - env.simulator.scene.env_origins[:, 2]
            self.max_box_height = max(self.max_box_height, float(heights[active].max().item()))
            self.min_root_height = min(self.min_root_height, float(roots[active, 2].min().item()))
        done = terminated | timeout
        end_frame = motion.motion.motion_end_idx[motion.motion_ids] - 1
        completed = (motion.time_steps >= end_frame) & ~done & active
        for index in completed.nonzero(as_tuple=False).flatten().tolist():
            self.clip_success[index] = True
        for index in (done & active).nonzero(as_tuple=False).flatten().tolist():
            self.counts[index] += 1
            row = {
                "env_id": index,
                "episode_index": self.counts[index],
                "duration_s": float(env.episode_length_buf[index].item() * env.dt),
                "motion_frame_at_termination": int(motion.time_steps[index].item()),
                "first_clip_completed": self.clip_success[index],
                "termination_reasons": [name for name, mask in components.items() if bool(mask[index].item())],
                "object_position_error_m": float(errors[index].item()),
                "object_orientation_error_rad": float(angles[index].item()),
            }
            self.episodes.append(row)
            self.clip_success[index] = False
            print("NATIVE_BOX_EPISODE", json.dumps(row), flush=True)
        return terminated, timeout

    def result(self):
        summary = summarize(self.episodes, len(self.counts) * self.goal)
        summary.update(
            control_steps=self.steps,
            active_state_samples=self.samples,
            object_position_rmse_before_reset_m=(self.position_squared_sum / self.samples) ** 0.5,
            object_mean_orientation_error_before_reset_rad=self.object_angle_sum / self.samples,
            box_max_center_height_m=self.max_box_height,
            robot_min_root_height_m=self.min_root_height,
        )
        return {"summary": summary, "episodes": self.episodes}


def worker(plan):
    import torch  # noqa: PLC0415
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config  # noqa: PLC0415
    from holosoma.utils.helpers import get_class  # noqa: PLC0415
    from holosoma.utils.rotations import quat_error_magnitude  # noqa: PLC0415
    from holosoma.utils.sim_utils import setup_simulation_environment, simulation_resource_session  # noqa: PLC0415
    from train_flux_box import check_task_config  # noqa: PLC0415

    checkpoint = plan["checkpoint"]
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    check_task_config(data["experiment_config"])
    assert int(data["iter"]) == 7999 and data["experiment_config"]["training"]["num_envs"] == 1024
    for name in (
        "actor_model_state_dict",
        "critic_model_state_dict",
        "actor_obs_normalizer_state_dict",
        "critic_obs_normalizer_state_dict",
    ):
        assert all(torch.isfinite(v).all() for v in data[name].values() if isinstance(v, torch.Tensor))
    saved, wandb_path = load_saved_experiment_config(CheckpointConfig(checkpoint=checkpoint))
    config = evaluation_config(saved, plan)
    output = Path(plan["result_dir"])
    config.save_config(str(output / "native_eval_config.yaml"))
    sys.argv = ["eval_flux_box.py"]
    env, device, app = setup_simulation_environment(config)
    with simulation_resource_session(env, app):
        env.set_is_evaluating()
        algo = get_class(config.algo._target_)(
            device=device,
            env=env,
            config=config.algo.config,
            log_dir=str(output),
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.attach_checkpoint_metadata(saved, wandb_path)
        algo.load(checkpoint)
        policy = algo.get_inference_policy()
        observer = NativeObserver(env, plan["episodes_per_env"], quat_error_magnitude)
        obs = env.reset_all()
        motion = env.command_manager.get_state("motion_command")
        print(
            "NATIVE_BOX_PROTOCOL",
            json.dumps(
                {
                    "num_envs": env.num_envs,
                    "episodes_per_env": plan["episodes_per_env"],
                    "control_dt": env.dt,
                    "motion_frames": int(motion.motion.time_step_total),
                    "eval_mode": env.is_evaluating,
                    "observation_noise": False,
                    "video": False,
                    "reset_phase": "native evaluation frame zero",
                    "startup_and_reset_randomization": "saved training configuration retained",
                    "random_pushes": "disabled by native evaluation mode",
                    "timeout_seconds": 10.0,
                    "runtime_versions": bootstrap.package_versions(),
                    "clip_success": (
                        "reach last reference frame without bad_tracking or timeout; native clip reset then occurs"
                    ),
                    "episode_counting": "fixed number per environment; simultaneous termination causes overlap",
                }
            ),
            flush=True,
        )
        observer.active = True
        with torch.inference_mode():
            for _ in range(plan["max_steps"]):
                actor_obs = torch.cat([obs[k] for k in algo.actor_obs_keys], dim=1)
                actions = policy({"actor_obs": actor_obs})
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Native policy returned non-finite actions")
                obs, _, _, _ = env.step({"actions": actions})
                if min(observer.counts) >= plan["episodes_per_env"]:
                    break
        result = observer.result()
        result["checkpoint_sha256"] = plan["checkpoint_sha256"]
        result["commit"] = plan["commit"]
        (output / "native_eval_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("NATIVE_BOX_EVAL_RESULT", json.dumps(result["summary"]), flush=True)
    print("NATIVE_BOX_WORKER_CLEAN_EXIT", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes-per-env", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--install-timeout", type=int, default=900)
    parser.add_argument("--evaluation-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.worker_plan:
        worker(json.loads(args.worker_plan.read_text(encoding="utf-8")))
        return
    if not 1 <= args.num_envs <= 64 or not 1 <= args.episodes_per_env <= 8 or not 500 <= args.max_steps <= 5000:
        parser.error("Evaluation supports 1..64 envs, 1..8 episodes/env, 500..5000 steps")
    if not 1 <= args.install_timeout <= 1800 or not 1 <= args.evaluation_timeout <= 1800:
        parser.error("Installation and evaluation timeouts must stay within 1..1800 seconds")
    env = bootstrap.prepare_environment()
    checkpoint = ROOT / "czy/data/exp_2.1/model_07999.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("Final checkpoint integrity mismatch")
    project = "flux-exp_2.1-native-eval-" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output = ROOT / "logs" / project
    output.mkdir(parents=True)
    plan = {
        "commit": repository_commit(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "project": project,
        "result_dir": str(output),
        **vars(args),
    }
    plan.pop("worker_plan")
    plan_path = output / "native_eval_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("FLUX_NATIVE_BOX_EVAL_PLAN", json.dumps(plan), flush=True)
    bootstrap.install_training_package(env, args.install_timeout)
    bootstrap.check_gpu(env)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-plan", str(plan_path)],
        check=True,
        timeout=args.evaluation_timeout,
        env=env,
    )
    result = json.loads((output / "native_eval_result.json").read_text(encoding="utf-8"))
    print("FLUX_NATIVE_BOX_EVAL_SUCCESS", json.dumps(result["summary"]), flush=True)
    time.sleep(60)


if __name__ == "__main__":
    main()
