"""Resume a verified box policy for additional PPO updates at 4096 envs.

A separate 4096-env/two-update capacity test runs by default. Formal training
reloads the original checkpoint, preserving both Adam optimizers and observation
normalizers. A previously validated configuration may explicitly skip the test.
No smoke weights are promoted into the formal run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import train_flux_box as box
from train_flux import repository_commit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "czy/data/exp_2.1/model_07999.pt"
SOURCE_SHA256 = "e2d120660bbc199d0dc03d0bce17a4989ef6117877c2f97f9da72d2c4dbc2c97"


def load_source(checkpoint=SOURCE, expected_sha256=SOURCE_SHA256, expected_iteration=7999):
    import torch  # noqa: PLC0415

    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError("Source checkpoint digest mismatch")
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    box.check_task_config(data["experiment_config"], allow_resume=True)
    assert int(data["iter"]) == expected_iteration
    for name in (
        "actor_model_state_dict",
        "critic_model_state_dict",
        "actor_obs_normalizer_state_dict",
        "critic_obs_normalizer_state_dict",
    ):
        assert data[name] and all(torch.isfinite(v).all() for v in data[name].values())
    for name in ("actor_optimizer_state_dict", "critic_optimizer_state_dict"):
        if not data[name]["state"]:
            raise RuntimeError("Full optimizer state is required for continuation")
    return data


def resume_config(source, project, additional_iterations, *, capacity=False, save_interval=100, checkpoint=SOURCE):
    from holosoma.config_types.experiment import ExperimentConfig  # noqa: PLC0415
    from holosoma.config_values.logger import disabled  # noqa: PLC0415
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object as native  # noqa: PLC0415

    saved = ExperimentConfig(**source["experiment_config"])
    saved_values = saved.to_serializable_dict()
    native_values = native.to_serializable_dict()
    saved_algo = dict(saved_values["algo"]["config"])
    native_algo = dict(native_values["algo"]["config"])
    for key in ("num_learning_iterations", "save_interval", "load_optimizer"):
        saved_algo.pop(key)
        native_algo.pop(key)
    # setup() resolves this symbolic dimension before checkpoint metadata is
    # saved. Compare that resolved value, without changing the runtime preset.
    if native_algo["module_dict"]["actor"]["output_dim"] == ["robot_action_dim"]:
        native_algo["module_dict"]["actor"]["output_dim"] = [len(native.robot.dof_names)]
    if saved_algo != native_algo or saved.algo._target_ != native.algo._target_:
        raise RuntimeError("Saved PPO configuration differs from native preset")
    for name in ("robot", "scene", "command", "action", "observation", "termination", "reward", "randomization"):
        if saved_values[name] != native_values[name]:
            raise RuntimeError(f"Saved {name} differs from native preset; review before continuation")
    return dataclasses.replace(
        native,
        training=dataclasses.replace(
            native.training, checkpoint=str(checkpoint), headless=True, project=project, name="g1_box_resume_ppo"
        ),
        logger=dataclasses.replace(
            disabled, video=dataclasses.replace(disabled.video, enabled=False), headless_recording=False
        ),
        algo=dataclasses.replace(
            native.algo,
            config=dataclasses.replace(
                native.algo.config,
                num_learning_iterations=additional_iterations,
                load_optimizer=True,
                save_interval=1 if capacity else save_interval,
            ),
        ),
    )


def state_equal(actual, expected):
    """Verify restoration on CPU so CUDA placement cannot hide missing state."""
    import torch  # noqa: PLC0415

    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and torch.equal(actual.detach().cpu(), expected.detach().cpu())
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(state_equal(actual[k], value) for k, value in expected.items())
        )
    if isinstance(expected, (tuple, list)):
        return (
            isinstance(actual, type(expected))
            and len(actual) == len(expected)
            and all(state_equal(a, b) for a, b in zip(actual, expected, strict=True))
        )
    return actual == expected


def restore(algo, source, checkpoint):
    algo.load(str(checkpoint))
    states = {
        "actor_model_state_dict": algo.actor,
        "critic_model_state_dict": algo.critic,
        "actor_obs_normalizer_state_dict": algo.actor_obs_normalizer,
        "critic_obs_normalizer_state_dict": algo.critic_obs_normalizer,
        "actor_optimizer_state_dict": algo.actor_optimizer,
        "critic_optimizer_state_dict": algo.critic_optimizer,
    }
    for name, module in states.items():
        if not state_equal(module.state_dict(), source[name]):
            raise RuntimeError(f"Resume restoration mismatch: {name}")
    # Native PPO.load stores the last completed index, while learn starts at
    # current_learning_iteration and performs num_learning_iterations updates.
    # Advance only this wrapper's runner so the last completed index is not repeated.
    algo.current_learning_iteration = int(source["iter"]) + 1
    algo.logging_helper.num_learning_iterations = algo.current_learning_iteration + algo.config.num_learning_iterations
    steps = [float(value["step"].item()) for value in source["actor_optimizer_state_dict"]["state"].values()]
    return {
        "source_iteration": int(source["iter"]),
        "next_iteration": algo.current_learning_iteration,
        "additional_iterations": algo.config.num_learning_iterations,
        "expected_final_iteration": algo.logging_helper.num_learning_iterations - 1,
        "weights_optimizers_normalizers_restored": True,
        "actor_optimizer_steps_min": min(steps),
        "actor_optimizer_steps_max": max(steps),
        "actor_learning_rate": algo.actor_learning_rate,
        "critic_learning_rate": algo.critic_learning_rate,
    }


def worker(plan):
    from holosoma.utils.helpers import get_class  # noqa: PLC0415
    from holosoma.utils.sim_utils import setup_simulation_environment, simulation_resource_session  # noqa: PLC0415

    checkpoint = Path(plan.get("source_checkpoint", SOURCE))
    digest = plan.get("source_sha256", SOURCE_SHA256)
    source = load_source(checkpoint, digest, plan.get("source_iteration", 7999))
    config = resume_config(
        source,
        plan["project"],
        plan["additional_iterations"],
        capacity=plan["capacity"],
        save_interval=plan.get("save_interval", 100),
        checkpoint=checkpoint,
    )
    output = Path(plan["result_dir"])
    config.save_config(str(output / "holosoma_config.yaml"))
    sys.argv = ["resume_flux_box.py"]
    env, device, app = setup_simulation_environment(config)
    with simulation_resource_session(env, app):
        algo = get_class(config.algo._target_)(
            device=device, env=env, config=config.algo.config, log_dir=str(output), multi_gpu_cfg=None
        )
        algo.setup()
        algo.attach_checkpoint_metadata(config)
        restored = restore(algo, source, checkpoint)
        restored.update(
            num_envs=env.num_envs,
            capacity=plan["capacity"],
            checkpoint_sha256=digest,
            seed=config.training.seed,
            save_interval=config.algo.config.save_interval,
        )
        (output / "resume_restored.json").write_text(json.dumps(restored, indent=2), encoding="utf-8")
        print("FLUX_BOX_RESUME_RESTORED", json.dumps(restored), flush=True)
        algo.learn()
        assert algo.current_learning_iteration == restored["expected_final_iteration"]
    print("FLUX_BOX_RESUME_WORKER_EXIT", json.dumps(restored), flush=True)


def verify(output, expected_iteration, updates, *, checkpoint=SOURCE, source_sha256=SOURCE_SHA256):
    import torch  # noqa: PLC0415

    result = box.bootstrap.verify_artifacts(output, expected_iteration + 1)
    saved = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    cfg = saved["experiment_config"]
    box.check_task_config(cfg, allow_resume=True)
    assert int(saved["iter"]) == expected_iteration
    assert cfg["training"]["num_envs"] == 4096 and cfg["training"]["checkpoint"] == str(checkpoint)
    assert cfg["algo"]["config"]["num_learning_iterations"] == updates
    for name in (
        "actor_model_state_dict",
        "critic_model_state_dict",
        "actor_obs_normalizer_state_dict",
        "critic_obs_normalizer_state_dict",
    ):
        assert all(torch.isfinite(v).all() for v in saved[name].values())
    result.update(
        source_sha256=source_sha256,
        num_envs=4096,
        additional_iterations=updates,
        final_iteration=expected_iteration,
        resume=True,
    )
    (output / "resume_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def stage(
    root,
    project,
    iterations,
    capacity,
    env,
    timeout,
    *,
    save_interval=100,
    checkpoint=SOURCE,
    checkpoint_sha256=SOURCE_SHA256,
    source_iteration=7999,
):
    output = root / ("capacity" if capacity else "formal")
    output.mkdir()
    plan = {
        "project": project,
        "result_dir": str(output),
        "additional_iterations": iterations,
        "capacity": capacity,
        "save_interval": save_interval,
        "source_checkpoint": str(checkpoint),
        "source_sha256": checkpoint_sha256,
        "source_iteration": source_iteration,
    }
    path = output / "worker_plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-plan", str(path)],
        env=env,
        check=True,
        timeout=timeout,
    )
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--verify-plan", str(path)], env=env, check=True, timeout=180
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=Path)
    parser.add_argument("--verify-plan", type=Path)
    parser.add_argument("--experiment", default="exp_2.2")
    parser.add_argument("--checkpoint", type=Path, default=SOURCE)
    parser.add_argument("--checkpoint-sha256", default=SOURCE_SHA256)
    parser.add_argument("--source-iteration", type=int, default=7999)
    parser.add_argument("--additional-iterations", type=int, default=6000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--skip-capacity-test", action="store_true", help="Reuse prior same-scale capacity validation")
    parser.add_argument("--install-timeout", type=int, default=900)
    parser.add_argument("--capacity-timeout", type=int, default=900)
    parser.add_argument("--training-timeout", type=int, default=86400)
    args = parser.parse_args()
    if args.worker_plan:
        worker(json.loads(args.worker_plan.read_text(encoding="utf-8")))
        return
    if args.verify_plan:
        plan = json.loads(args.verify_plan.read_text(encoding="utf-8"))
        result = verify(
            Path(plan["result_dir"]),
            plan.get("source_iteration", 7999) + plan["additional_iterations"],
            plan["additional_iterations"],
            checkpoint=Path(plan.get("source_checkpoint", SOURCE)),
            source_sha256=plan.get("source_sha256", SOURCE_SHA256),
        )
        marker = "FLUX_BOX_CAPACITY_SUCCESS" if plan["capacity"] else "FLUX_BOX_RESUME_SUCCESS"
        print(marker, json.dumps(result), flush=True)
        return
    if not re.fullmatch(r"exp_[0-9]+(?:\.[0-9]+)?", args.experiment):
        parser.error("Invalid experiment identifier")
    if not 1 <= args.additional_iterations <= 30000 or not 1 <= args.training_timeout <= 86400:
        parser.error("Additional updates must be 1..30000; training timeout must be 1..86400 seconds")
    if not 1 <= args.save_interval <= args.additional_iterations:
        parser.error("Save interval must be 1..additional updates")
    if not 1 <= args.install_timeout <= 1800 or not 1 <= args.capacity_timeout <= 1800:
        parser.error("Installation and capacity limits must be 1..1800 seconds")
    if args.source_iteration < 0 or not re.fullmatch(r"[a-f0-9]{64}", args.checkpoint_sha256):
        parser.error("Source iteration must be non-negative and digest must be lowercase SHA-256")
    env = box.bootstrap.prepare_environment()
    checkpoint = args.checkpoint.resolve()
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
        raise RuntimeError("Source checkpoint digest mismatch before installation")
    project = f"flux-{args.experiment}-box-resume-" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    root = (ROOT / "logs" / project).resolve()
    root.mkdir(parents=True)
    plan = {
        "commit": repository_commit(),
        "source_checkpoint": str(checkpoint),
        "source_sha256": args.checkpoint_sha256,
        "source_iteration": args.source_iteration,
        "next_iteration": args.source_iteration + 1,
        "expected_final_iteration": args.source_iteration + args.additional_iterations,
        "num_envs": 4096,
        "capacity_iterations": 0 if args.skip_capacity_test else 2,
        **vars(args),
    }
    plan.pop("worker_plan")
    plan.pop("verify_plan")
    plan["checkpoint"] = str(checkpoint)
    (root / "launch_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("FLUX_BOX_RESUME_PLAN", json.dumps(plan), flush=True)
    box.bootstrap.install_training_package(env, args.install_timeout)
    box.bootstrap.check_gpu(env)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    source_args = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": args.checkpoint_sha256,
        "source_iteration": args.source_iteration,
    }
    if args.skip_capacity_test:
        print("FLUX_BOX_CAPACITY_SKIPPED prior same-scale validation reused", flush=True)
    else:
        print("FLUX_BOX_CAPACITY_START 4096 envs, 2 updates", flush=True)
        stage(root, project + "-capacity", 2, True, env, args.capacity_timeout, **source_args)
    print("FLUX_BOX_RESUME_FORMAL_START", project, flush=True)
    stage(
        root,
        project,
        args.additional_iterations,
        False,
        env,
        args.training_timeout,
        save_interval=args.save_interval,
        **source_args,
    )
    time.sleep(60)


if __name__ == "__main__":
    main()
