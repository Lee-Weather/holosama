"""Replay this project's trained G1 PPO policy in native Windows MuJoCo.

Uses the checkpoint's alphabetical observation layout, joint names, ELU MLP,
PD gains and simulation clock. No robot SDK, DDS, IsaacSim, or training install.
W/S: forward speed; A/D: lateral speed; Q/E: yaw; X: stop; R: reset;
Space: pause. The first 30 simulated seconds are recorded when --video is set.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import time
from contextlib import ExitStack
from pathlib import Path
from queue import Empty, SimpleQueue

import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "czy/data/exp_1.1/model_04999.pt"


def load_actor(checkpoint: Path):
    """Reconstruct only the supported plain PPO actor, with strict state loading."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["experiment_config"]
    actor_cfg = cfg["algo"]["config"]["module_dict"]["actor"]
    layer = actor_cfg["layer_config"]
    if (
        actor_cfg["type"] != "MLP"
        or layer["activation"] != "ELU"
        or layer["use_layer_norm"]
        or layer["dropout_prob"]
        or cfg["algo"]["config"]["empirical_normalization"]
    ):
        raise ValueError("This replay supports unnormalized plain ELU PPO MLP checkpoints only")
    if cfg["observation"]["groups"]["actor_obs"]["history_length"] != 1:
        raise ValueError("Observation history is not supported by this replay")
    dimensions = [saved["obs_dim_dict"]["actor_obs"], *layer["hidden_dims"], cfg["robot"]["actions_dim"]]
    layers = []
    for index, (size_in, size_out) in enumerate(itertools.pairwise(dimensions)):
        layers.append(torch.nn.Linear(size_in, size_out))
        if index < len(dimensions) - 2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    state = saved["actor_model_state_dict"]
    expected = {"actor_module.module." + key for key in actor.state_dict()} | {"std"}
    if set(state) != expected or not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Actor state keys or weights are invalid")
    actor.load_state_dict(
        {key.removeprefix("actor_module.module."): value for key, value in state.items() if key != "std"}
    )
    actor.eval()
    return actor, cfg, int(saved["iter"])


def export_actor(actor, cfg: dict, checkpoint: Path, destination: Path) -> dict:
    """Export the trained actor and its configuration for local deployment."""
    import onnx  # noqa: PLC0415

    example = torch.zeros(1, actor[0].in_features)
    torch.onnx.export(
        actor,
        example,
        str(destination),
        input_names=["actor_obs"],
        output_names=["action"],
        opset_version=13,
        **({"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}),
    )
    model = onnx.load(str(destination))
    for key, value in {
        "experiment_config": cfg,
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "observation_term_order": sorted(cfg["observation"]["groups"]["actor_obs"]["terms"]),
    }.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, json.dumps(value)
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    return {"path": str(destination)}


class PolicyReplay:
    def __init__(self, checkpoint: Path, velocity: list[float]):
        torch.set_num_threads(1)
        self.actor, self.cfg, self.iteration = load_actor(checkpoint)
        robot = self.cfg["robot"]
        sim = self.cfg["simulator"]["config"]["sim"]
        scene = ROOT / "src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = 1 / sim["fps"]
        self.decimation = sim["control_decimation"]
        self.dt = self.model.opt.timestep * self.decimation
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 1280, 720
        names = robot["dof_names"]
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
        if min([*joint_ids, *self.actuator_ids]) < 0 or self.model.nu != len(names):
            raise ValueError("MJCF joints/actuators do not match the checkpoint")
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[joint_ids]
        self.default_pos = np.array([robot["init_state"]["default_joint_angles"][name] for name in names])
        self.control = robot["control"]
        self.kp, self.kd = [], []
        for name in names:
            matches = [key for key in self.control["stiffness"] if key in name]
            if not matches:
                raise ValueError(f"Missing PD gains for {name}")
            key = matches[-1]
            self.kp.append(self.control["stiffness"][key])
            self.kd.append(self.control["damping"][key])
        self.kp, self.kd = np.array(self.kp), np.array(self.kd)
        self.scale = np.full(len(names), self.control["action_scale"])
        self.torque_limits = np.array(robot["dof_effort_limit_list"])
        if self.control["action_scales_by_effort_limit_over_p_gain"]:
            self.scale *= self.torque_limits / self.kp
        # Match the configured joint armatures/friction instead of silently
        # retaining different defaults from the MJCF.
        self.model.dof_armature[self.qvel_ids] = robot["dof_armature_list"]
        self.model.dof_frictionloss[self.qvel_ids] = robot["dof_joint_friction_list"]
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.data = mujoco.MjData(self.model)
        self.velocity = np.array(velocity, dtype=float)
        self.period = self.cfg["command"]["setup_terms"]["locomotion_gait"]["params"]["gait_period"]
        self.total_steps = 0
        self.falls = []
        self.episode_steps = 0
        self.keys = SimpleQueue()
        self.paused = False
        self.last_action = np.zeros(len(names))
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        initial = self.cfg["robot"]["init_state"]
        self.data.qpos[:3] = initial["pos"]
        xyzw = initial["rot"]
        self.data.qpos[3:7] = [xyzw[3], *xyzw[:3]]
        self.data.qpos[self.qpos_ids] = self.default_pos
        self.last_action.fill(0)
        self.episode_steps = 0
        mujoco.mj_forward(self.model, self.data)

    def observation(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        body_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, body_velocity, 1)
        phase = self.episode_steps * self.dt * 2 * np.pi / self.period + np.array([0, -np.pi])
        if np.linalg.norm(self.velocity[:2]) < 0.01 and abs(self.velocity[2]) < 0.01:
            phase[:] = np.pi
        values = {
            "actions": self.last_action,
            "base_ang_vel": body_velocity[:3],
            "command_ang_vel": self.velocity[2:],
            "command_lin_vel": self.velocity[:2],
            "cos_phase": np.cos(phase),
            "dof_pos": self.data.qpos[self.qpos_ids] - self.default_pos,
            "dof_vel": self.data.qvel[self.qvel_ids],
            "projected_gravity": rotation.T @ np.array([0, 0, -1]),
            "sin_phase": np.sin(phase),
        }
        terms = self.cfg["observation"]["groups"]["actor_obs"]["terms"]
        if set(terms) != set(values):
            raise ValueError("Unsupported actor observation terms")
        packed = []
        for key in sorted(terms):
            term = terms[key]
            value = values[key] * term["scale"]
            if term["clip"] is not None:
                value = np.clip(value, *term["clip"])
            if term["global_clip"]:
                value = np.clip(
                    value, -self.cfg["observation"]["clip_observations"], self.cfg["observation"]["clip_observations"]
                )
            packed.append(value)
        observation = np.concatenate(packed).astype(np.float32)
        if observation.shape != (self.actor[0].in_features,) or not np.isfinite(observation).all():
            raise RuntimeError("Invalid observation")
        return observation

    def step(self):
        with torch.inference_mode():
            action = self.actor(torch.from_numpy(self.observation()).unsqueeze(0)).numpy()[0]
        if not np.isfinite(action).all():
            raise RuntimeError("Nonfinite policy action")
        if self.control["clip_actions"]:
            action = np.clip(action, -self.control["action_clip_value"], self.control["action_clip_value"])
        target = self.default_pos + action * self.scale
        for _ in range(self.decimation):
            torque = self.kp * (target - self.data.qpos[self.qpos_ids]) - self.kd * self.data.qvel[self.qvel_ids]
            if self.control["clip_torques"]:
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl[self.actuator_ids] = torque
            mujoco.mj_step(self.model, self.data)
        self.last_action[:] = action
        self.episode_steps += 1
        self.total_steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulated state")
        # Local replay fall detector; not claimed to reproduce IsaacSim's
        # contact-based training termination statistics.
        if self.data.qpos[2] < 0.35 or self.data.xmat[self.base_id].reshape(3, 3)[2, 2] < 0.4:
            self.falls.append({"time_s": self.total_steps * self.dt, "episode_s": self.episode_steps * self.dt})
            self.reset()
        elif self.episode_steps * self.dt >= 20:
            self.reset()

    def handle_keys(self):
        while True:
            try:
                key = self.keys.get_nowait()
            except Empty:
                return
            if key == 32:
                self.paused = not self.paused
            elif key == ord("R"):
                self.reset()
            elif key == ord("X"):
                self.velocity[:] = 0
            elif key in (ord("W"), ord("S"), ord("A"), ord("D"), ord("Q"), ord("E")):
                axis, delta = {
                    ord("W"): (0, 0.1),
                    ord("S"): (0, -0.1),
                    ord("A"): (1, 0.1),
                    ord("D"): (1, -0.1),
                    ord("Q"): (2, 0.1),
                    ord("E"): (2, -0.1),
                }[key]
                self.velocity[axis] = np.clip(self.velocity[axis] + delta, -1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--velocity", nargs=3, type=float, default=[0.3, 0, 0])
    parser.add_argument("--video", type=Path)
    parser.add_argument("--record-seconds", type=float, default=30)
    parser.add_argument("--export-onnx", type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600 or not 0 < args.record_seconds <= 120:
        parser.error("Playback must be 0..3600s; recording must be 0..120s")
    replay = PolicyReplay(args.checkpoint, args.velocity)
    if args.export_onnx:
        print(
            "ONNX_EXPORT",
            json.dumps(export_actor(replay.actor, replay.cfg, args.checkpoint, args.export_onnx)),
            flush=True,
        )
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.5, 130, -18
    print(
        "LOCAL_REPLAY_STARTED",
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "iteration": replay.iteration,
                "mujoco": mujoco.__version__,
                "physics_dt": replay.model.opt.timestep,
                "control_dt": replay.dt,
                "command": args.velocity,
                "observation_order": sorted(replay.cfg["observation"]["groups"]["actor_obs"]["terms"]),
            }
        ),
        flush=True,
    )
    frames = 0
    with ExitStack() as stack:
        viewer = (
            None
            if args.headless
            else stack.enter_context(
                mujoco.viewer.launch_passive(
                    replay.model, replay.data, key_callback=replay.keys.put, show_left_ui=False, show_right_ui=False
                )
            )
        )
        renderer = writer = None
        if args.video:
            import cv2  # noqa: PLC0415

            args.video.parent.mkdir(parents=True, exist_ok=True)
            renderer = stack.enter_context(mujoco.Renderer(replay.model, height=720, width=1280))
            writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
            if not writer.isOpened():
                raise RuntimeError("Video writer did not open")
            stack.callback(writer.release)
        started = time.perf_counter()
        try:
            while replay.total_steps * replay.dt < args.seconds and (viewer is None or viewer.is_running()):
                tick = time.perf_counter()
                replay.handle_keys()
                if not replay.paused:
                    replay.step()
                camera.lookat[:] = replay.data.qpos[:3]
                camera.lookat[2] = 0.65
                if viewer:
                    viewer.cam.lookat[:] = camera.lookat
                    if replay.total_steps <= 1:
                        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                            camera.distance,
                            camera.azimuth,
                            camera.elevation,
                        )
                    viewer.sync()
                sim_time = replay.total_steps * replay.dt
                if (
                    writer
                    and not replay.paused
                    and frames < round(args.record_seconds * 30)
                    and sim_time >= frames / 30
                ):
                    renderer.update_scene(replay.data, camera=camera)
                    writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
                    frames += 1
                    if frames == round(args.record_seconds * 30):
                        writer.release()
                        print("LOCAL_VIDEO_SAVED", str(args.video), flush=True)
                        writer = None
                if viewer:
                    time.sleep(max(0, replay.dt - (time.perf_counter() - tick)))
        finally:
            report = {
                "checkpoint": str(args.checkpoint),
                "iteration": replay.iteration,
                "mujoco": mujoco.__version__,
                "control_steps": replay.total_steps,
                "simulated_seconds": replay.total_steps * replay.dt,
                "wall_seconds": time.perf_counter() - started,
                "fall_count": len(replay.falls),
                "falls": replay.falls,
                "recorded_frames": frames,
                "video": str(args.video) if args.video else None,
                "finite_state": bool(np.isfinite(replay.data.qpos).all()),
            }
            output = args.video.with_suffix(".json") if args.video else ROOT / "logs/local_mujoco_replay.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("LOCAL_REPLAY_RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
