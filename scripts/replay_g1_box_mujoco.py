"""Record a learned G1 box-carry policy in a free-body MuJoCo simulation.

The training URDF supplies collision geometry. Only the initial robot/box states
come from motion data; thereafter the actor and PD torques drive mj_step.
No trajectory playback, object pose correction, or reset hides failed tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def quaternion_matrix(quaternion):
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


class BoxPolicy:
    def __init__(self, checkpoint: Path, mass: float, without_box: bool = False):
        torch.set_num_threads(1)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = saved["experiment_config"]
        import train_flux_box  # noqa: PLC0415

        train_flux_box.check_task_config(self.cfg)
        self.iteration = int(saved["iter"])
        self.checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        robot = self.cfg["robot"]
        self.names = robot["dof_names"]
        actor_cfg = self.cfg["algo"]["config"]["module_dict"]["actor"]
        layer = actor_cfg["layer_config"]
        if layer["activation"] != "ELU" or layer["use_layer_norm"] or layer["dropout_prob"]:
            raise ValueError("Unsupported actor architecture")
        dimensions = [saved["obs_dim_dict"]["actor_obs"], *layer["hidden_dims"], len(self.names)]
        layers = []
        for index in range(len(dimensions) - 1):
            layers.append(torch.nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(torch.nn.ELU())
        self.actor = torch.nn.Sequential(*layers)
        state = saved["actor_model_state_dict"]
        assert set(state) == {"std"} | {"actor_module.module." + k for k in self.actor.state_dict()}
        self.actor.load_state_dict({k.removeprefix("actor_module.module."): v for k, v in state.items() if k != "std"})
        self.actor.eval()
        assert self.cfg["algo"]["config"]["empirical_normalization"]
        normalizer = saved["actor_obs_normalizer_state_dict"]
        self.obs_mean = normalizer["_mean"]
        self.obs_std = normalizer["_std"]
        assert torch.isfinite(self.obs_mean).all() and torch.isfinite(self.obs_std).all()
        motion_cfg = self.cfg["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
        motion_path = ROOT / "src/holosoma" / motion_cfg["motion_file"]
        if hashlib.sha256(motion_path.read_bytes()).hexdigest() != train_flux_box.MOTION_SHA256:
            raise ValueError("Motion differs from the trained reference")
        with np.load(motion_path, allow_pickle=False) as data:
            self.motion = {k: data[k].copy() for k in data.files}
        motion_joint_names = self.motion["joint_names"].tolist()
        self.motion_joint_ids = [motion_joint_names.index(name) for name in self.names]
        body_names = self.motion["body_names"].tolist()
        self.motion_root_id = body_names.index("pelvis")
        self.ref_name = motion_cfg["body_name_ref"][0]
        self.motion_ref_id = body_names.index(self.ref_name)
        self.frames = len(self.motion["joint_pos"])
        self.motion_fps = float(self.motion["fps"].item())
        self.control = robot["control"]
        self.default = np.array([robot["init_state"]["default_joint_angles"][name] for name in self.names])
        self.kp = np.array(
            [next(value for key, value in self.control["stiffness"].items() if key in name) for name in self.names]
        )
        self.kd = np.array(
            [next(value for key, value in self.control["damping"].items() if key in name) for name in self.names]
        )
        self.limits = np.array(robot["dof_effort_limit_list"])
        self.scale = np.full(len(self.names), self.control["action_scale"])
        if self.control["action_scales_by_effort_limit_over_p_gain"]:
            self.scale *= self.limits / self.kp
        urdf = ROOT / "src/holosoma/holosoma/data/robots" / robot["asset"]["urdf_file"]
        spec = mujoco.MjSpec.from_file(str(urdf))
        spec.body("pelvis").add_freejoint(name="robot_free")
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.1],
            rgba=[0.22, 0.28, 0.35, 1],
            friction=[1, 0.005, 0.0001],
        )
        spec.worldbody.add_light(pos=[1, -2, 4], dir=[-0.2, 0.2, -1])
        if not without_box:
            box_mesh = ROOT / "src/holosoma/holosoma/data/scene_objects/boxes/largebox.obj"
            spec.add_mesh(name="carry_box_mesh", file=str(box_mesh))
            box = spec.worldbody.add_body(name="carry_box")
            box.add_freejoint(name="box_free")
            box.add_geom(
                name="carry_box_collision",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="carry_box_mesh",
                mass=mass,
                rgba=[0.85, 0.55, 0.25, 1],
                friction=[0.35, 0.005, 0.0001],
            )
        for name, limit in zip(self.names, self.limits, strict=True):
            spec.add_actuator(
                name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrllimited=True, ctrlrange=[-limit, limit]
            )
        self.model = spec.compile()
        sim = self.cfg["simulator"]["config"]["sim"]
        self.model.opt.timestep = 1 / sim["fps"]
        self.decimation = sim["control_decimation"]
        self.dt = self.model.opt.timestep * self.decimation
        if abs(self.dt * self.motion_fps - 1) > 1e-8:
            raise ValueError("Motion and policy clocks differ")
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 1280, 720
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.names]
        if min(joint_ids) < 0:
            raise ValueError("Training joints not found in the imported URDF")
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[joint_ids]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.names]
        self.model.dof_armature[self.qvel_ids] = robot["dof_armature_list"]
        self.model.dof_frictionloss[self.qvel_ids] = robot["dof_joint_friction_list"]
        self.root_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.ref_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ref_name)
        self.without_box = without_box
        self.box_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "carry_box")
        if without_box:
            assert self.box_body == -1 and self.model.nq == 7 + len(self.names)
        else:
            box_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_free")
            self.box_pos_id = self.model.jnt_qposadr[box_joint]
            self.box_vel_id = self.model.jnt_dofadr[box_joint]
            # Preserve the training URDF's inertia-to-mass ratio at nominal mass.
            self.model.body_mass[self.box_body] = mass
            self.model.body_inertia[self.box_body] = 0.002 * mass / 0.1
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:7] = self.motion["joint_pos"][0, :7]
        self.data.qpos[self.qpos_ids] = self.motion["joint_pos"][0, 7:][self.motion_joint_ids]
        # The NPZ stores native MuJoCo free-joint velocities: world-frame
        # origin linear velocity, followed by local-frame angular velocity.
        # body_lin_vel_w is measured at the COM and cannot be copied here.
        self.data.qvel[:6] = self.motion["joint_vel"][0, :6]
        self.data.qvel[self.qvel_ids] = self.motion["joint_vel"][0, 6:][self.motion_joint_ids]
        if not without_box:
            self.data.qpos[self.box_pos_id : self.box_pos_id + 3] = self.motion["object_pos_w"][0]
            self.data.qpos[self.box_pos_id + 3 : self.box_pos_id + 7] = self.motion["object_quat_w"][0]
            self.data.qvel[self.box_vel_id : self.box_vel_id + 3] = self.motion["object_lin_vel_w"][0]
        mujoco.mj_forward(self.model, self.data)
        self.action = np.zeros(len(self.names))
        self.index = 0
        self.metrics = []
        self.box_mass = mass
        self.initial_alignment = {
            "root_velocity_source": "motion.joint_vel[0,:6]: world origin linear / local angular",
            "root_position_difference_m": float(
                np.linalg.norm(self.data.xpos[self.root_body] - self.motion["body_pos_w"][0, self.motion_root_id])
            ),
            "torso_position_difference_m": float(
                np.linalg.norm(self.data.xpos[self.ref_body] - self.motion["body_pos_w"][0, self.motion_ref_id])
            ),
            "collision_geometries": int(np.count_nonzero(self.model.geom_contype)),
            "unit_motor_gears": bool(np.all(self.model.actuator_gear[:, 0] == 1)),
        }

    def observation(self):
        index = min(self.index, self.frames - 1)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.root_body, velocity, 1)
        rotation = self.data.xmat[self.ref_body].reshape(3, 3)
        reference = quaternion_matrix(self.motion["body_quat_w"][index, self.motion_ref_id])
        values = {
            "actions": self.action,
            "base_ang_vel": velocity[:3],
            "dof_pos": self.data.qpos[self.qpos_ids] - self.default,
            "dof_vel": self.data.qvel[self.qvel_ids],
            "motion_command": np.concatenate(
                (
                    self.motion["joint_pos"][index, 7:][self.motion_joint_ids],
                    self.motion["joint_vel"][index, 6:][self.motion_joint_ids],
                )
            ),
            "motion_ref_ori_b": (rotation.T @ reference)[:, :2].reshape(-1),
        }
        group = self.cfg["observation"]["groups"]["actor_obs"]
        assert group["history_length"] == 1 and set(values) == set(group["terms"])
        packed = []
        for key in sorted(group["terms"]):
            term = group["terms"][key]
            value = values[key] * term["scale"]
            if term["clip"] is not None:
                value = np.clip(value, *term["clip"])
            if term["global_clip"]:
                limit = self.cfg["observation"]["clip_observations"]
                value = np.clip(value, -limit, limit)
            packed.append(value)
        observation = torch.from_numpy(np.concatenate(packed).astype(np.float32)).unsqueeze(0)
        return (observation - self.obs_mean) / (self.obs_std + 1e-2)

    def record_metrics(self):
        metric = {
            "time_s": self.index * self.dt,
            "robot_height_m": float(self.data.qpos[2]),
        }
        if self.without_box:
            self.metrics.append(metric)
            return
        index = min(self.index, self.frames - 1)
        position = self.data.xpos[self.box_body].copy()
        actual_rotation = self.data.xmat[self.box_body].reshape(3, 3)
        reference_rotation = quaternion_matrix(self.motion["object_quat_w"][index])
        angle = np.arccos(np.clip((np.trace(actual_rotation.T @ reference_rotation) - 1) / 2, -1, 1))
        metric.update(
            {
                "box_position": position.tolist(),
                "box_position_error_m": float(np.linalg.norm(position - self.motion["object_pos_w"][index])),
                "box_orientation_error_rad": float(angle),
            }
        )
        self.metrics.append(metric)

    def step(self):
        with torch.inference_mode():
            action = self.actor(self.observation()).numpy()[0]
        if not np.isfinite(action).all():
            raise RuntimeError("Policy actions are not finite")
        if self.control["clip_actions"]:
            action = np.clip(action, -self.control["action_clip_value"], self.control["action_clip_value"])
        target = self.default + self.scale * action
        for _ in range(self.decimation):
            torque = self.kp * (target - self.data.qpos[self.qpos_ids]) - self.kd * self.data.qvel[self.qvel_ids]
            if self.control["clip_torques"]:
                torque = np.clip(torque, -self.limits, self.limits)
            self.data.ctrl[self.actuator_ids] = torque
            mujoco.mj_step(self.model, self.data)
        self.action[:] = action
        self.index += 1
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Simulation state is not finite")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "czy/data/exp_2.1/model_07999.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--no-box", action="store_true", help="Remove the box body, collision geometry and rendering; keep the actor."
    )
    parser.add_argument("--mass", type=float, default=2.6)
    args = parser.parse_args()
    stem = "robot_only" if args.no_box else "box"
    args.output = args.output or ROOT / f"czy/data/exp_2.1/play_{stem}_mujoco.mp4"
    args.report = args.report or ROOT / f"czy/exp1/artifacts/exp_2.1/local_{stem}_replay_validation.json"
    if not 1.1 <= args.mass <= 4.1:
        parser.error("Box mass must stay within the training range 1.1..4.1 kg")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    replay = BoxPolicy(args.checkpoint, args.mass, without_box=args.no_box)
    raw = args.output.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 50, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("Video writer could not open")
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.0, 135, -18
    print(
        "BOX_REPLAY_START",
        json.dumps({"iteration": replay.iteration, "frames": replay.frames, "box_removed": args.no_box}),
        flush=True,
    )
    try:
        with mujoco.Renderer(replay.model, height=720, width=1280) as renderer:
            for _ in range(replay.frames):
                replay.record_metrics()
                camera.lookat[:] = replay.data.xpos[replay.root_body]
                camera.lookat[2] = 0.6
                renderer.update_scene(replay.data, camera=camera)
                rgb = renderer.render()
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                scene_label = "NO BOX | learned policy" if args.no_box else f"dynamic box {args.mass:g} kg"
                cv2.putText(
                    frame,
                    f"{args.checkpoint.parent.name} | model {replay.iteration:05d} | "
                    f"{scene_label} | t={replay.index * replay.dt:.2f}s",
                    (25, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                writer.write(frame)
                replay.step()
    finally:
        writer.release()
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-y",
            "-i",
            str(raw),
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
            str(args.output),
        ],
        check=True,
        timeout=120,
        capture_output=True,
    )
    raw.unlink()
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": replay.checkpoint_hash,
        "iteration": replay.iteration,
        "mujoco": mujoco.__version__,
        "box_removed": args.no_box,
        "box_body_count": int(replay.box_body >= 0),
        "box_mass_kg": None if args.no_box else args.mass,
        "box_sliding_friction": None if args.no_box else 0.35,
        "frames": replay.frames,
        "duration_s": replay.frames * replay.dt,
        "control_dt": replay.dt,
        "initial_alignment": replay.initial_alignment,
        "automatic_resets": 0,
        "pose_corrections_after_initialization": 0,
        "robot_min_height_m": min(v["robot_height_m"] for v in replay.metrics),
        "video_bytes": args.output.stat().st_size,
        "video_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "replay_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": (
            "Cross-simulator evaluation using training robot collision URDF. "
            "This is an independent physical rollout, not reference motion playback."
        ),
    }
    if not args.no_box:
        position_errors = [v["box_position_error_m"] for v in replay.metrics]
        orientation_errors = [v["box_orientation_error_rad"] for v in replay.metrics]
        violations = [
            v for v in replay.metrics if v["box_position_error_m"] > 0.25 or v["box_orientation_error_rad"] > 0.8
        ]
        result.update(
            box_position_rmse_m=float(np.sqrt(np.mean(np.square(position_errors)))),
            box_position_max_error_m=max(position_errors),
            box_orientation_max_error_rad=max(orientation_errors),
            first_box_tracking_violation_s=violations[0]["time_s"] if violations else None,
            box_max_height_m=max(v["box_position"][2] for v in replay.metrics),
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": result, "trajectory": replay.metrics}, indent=2), encoding="utf-8")
    print("BOX_REPLAY_RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
