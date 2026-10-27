"""Replay bundled G1 reference motions with MuJoCo, without training dependencies.

Run: python scripts/replay_g1_reference_mujoco.py
Record MP4s: python scripts/replay_g1_reference_mujoco.py --record-dir videos
Keys: 1/2/3 select motion, Space pause, R restart, Left/Right seek,
Up/Down change speed, A toggle automatic advance. Close the viewer to exit.
"""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path
from queue import Empty, SimpleQueue

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
MODEL_DIR = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1"
CLIPS = {
    "robot": ("sub3_largebox_003_mj.npz", "Body tracking (no box)"),
    "object": ("sub3_largebox_003_mj_w_obj.npz", "Box lifting and carrying"),
    "slope": ("motion_crawl_slope.npz", "Slope crawling"),
}


def build_model() -> mujoco.MjModel:
    """Use the original conversion model so forward kinematics match the NPZ."""
    tree = ET.parse(MODEL_DIR / "g1_29dof_w_largebox.xml")  # noqa: S314 -- fixed repository asset
    root = tree.getroot()
    root.find("compiler").set("meshdir", (MODEL_DIR / "assets").as_posix())
    root.set("model", "Holosoma G1 reference replay")
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "material",
        name="replay_terrain_mat",
        rgba="0.7 0.85 0.6 1",
        texture="texplane",
        texuniform="true",
        texrepeat="2 2",
        specular="0",
        shininess="0",
    )
    for material in root.findall("asset/material"):
        if material.get("name") == "MatPlane":
            material.set("reflectance", "0")
    for light in root.findall("worldbody/light"):
        light.set("pos", "3 -4 7")
        light.set("directional", "true")
        light.set("diffuse", "0.6 0.6 0.6")
        light.set("specular", "0 0 0")
    ET.SubElement(asset, "mesh", name="replay_slope_mesh", file=(MOTION_DIR / "terrain_slope.obj").as_posix())
    ET.SubElement(
        root.find("worldbody"),
        "geom",
        name="replay_slope",
        type="mesh",
        mesh="replay_slope_mesh",
        rgba="0.55 0.65 0.48 1",
        material="replay_terrain_mat",
        group="2",
        contype="0",
        conaffinity="0",
    )
    visual = root.find("visual")
    visual.find("headlight").set("specular", "0 0 0")
    visual.find("headlight").set("ambient", "0.3 0.3 0.3")
    visual.find("global").set("offwidth", "1280")
    visual.find("global").set("offheight", "720")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # Playback writes recorded states and runs forward kinematics, never mj_step.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    return model


class ReferencePlayer:
    def __init__(self, selected: str, speed: float):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.clips = {}
        for key, (filename, _) in CLIPS.items():
            with np.load(MOTION_DIR / filename, allow_pickle=False) as source:
                self.clips[key] = {name: source[name] for name in source.files}
        self.keys = list(CLIPS)
        self.speed = speed
        self.paused = False
        self.auto_advance = selected == "all"
        self.cursor = 0.0
        self.pending_keys: SimpleQueue[int] = SimpleQueue()
        self.box_geoms = np.flatnonzero(self.model.geom_bodyid == self.model.body("largebox_link").id)
        self.slope_geom = self.model.geom("replay_slope").id
        self.floor_geom = self.model.geom("ground").id
        self.root_joint = int(self.model.body("pelvis").jntadr[0])
        self.box_joint = int(self.model.body("largebox_link").jntadr[0])
        self.select("robot" if selected == "all" else selected)

    def select(self, key: str) -> None:
        self.key = key
        self.clip = self.clips[key]
        self.cursor = 0.0
        self.fps = float(self.clip["fps"].item())
        self.frames = len(self.clip["joint_pos"])
        self.joint_ids = [self.model.joint(str(name)).id for name in self.clip["joint_names"]]
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_indices = self.model.jnt_dofadr[self.joint_ids]
        self.body_ids = [self.model.body(str(name)).id for name in self.clip["body_names"]]
        self.model.geom_group[self.box_geoms] = 1 if key == "object" else 4
        self.model.geom_group[self.slope_geom] = 2 if key == "slope" else 4
        self.model.geom_group[self.floor_geom] = 4 if key == "slope" else 2
        self.apply_frame(0)
        print(f"Playing {CLIPS[key][0]}: {self.frames} frames, {self.fps:g} Hz", flush=True)

    def apply_frame(self, frame: int) -> None:
        clip, model, data = self.clip, self.model, self.data
        root_q = int(model.jnt_qposadr[self.root_joint])
        root_v = int(model.jnt_dofadr[self.root_joint])
        box_q = int(model.jnt_qposadr[self.box_joint])
        box_v = int(model.jnt_dofadr[self.box_joint])
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0
        data.qpos[root_q : root_q + 7] = clip["joint_pos"][frame, :7]
        data.qvel[root_v : root_v + 6] = clip["joint_vel"][frame, :6]
        data.qpos[self.qpos_indices] = clip["joint_pos"][frame, 7:]
        data.qvel[self.qvel_indices] = clip["joint_vel"][frame, 6:]
        if "object_pos_w" in clip:
            data.qpos[box_q : box_q + 3] = clip["object_pos_w"][frame]
            data.qpos[box_q + 3 : box_q + 7] = clip["object_quat_w"][frame]
            data.qvel[box_v : box_v + 3] = clip["object_lin_vel_w"][frame]
            data.qvel[box_v + 3 : box_v + 6] = clip["object_ang_vel_w"][frame]
        else:
            data.qpos[box_q : box_q + 3] = [0, 0, -10]
        data.time = frame / self.fps
        mujoco.mj_forward(model, data)

    def camera(self, camera: mujoco.MjvCamera) -> None:
        xyz = self.clip["joint_pos"][:, :3]
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        camera.lookat[2] = 0.65 if self.key != "slope" else 0.6
        camera.distance = 3.3 if self.key != "slope" else 4.6
        camera.azimuth = 145
        camera.elevation = -22

    def handle_keys(self) -> bool:
        """Consume viewer-thread events on the playback thread."""
        changed = False
        while True:
            try:
                key = self.pending_keys.get_nowait()
            except Empty:
                return changed
            if key in (ord("1"), ord("2"), ord("3")):
                self.select(self.keys[key - ord("1")])
                changed = True
            elif key == ord(" "):
                self.paused = not self.paused
            elif key == ord("R"):
                self.cursor = 0.0
            elif key == ord("A"):
                self.auto_advance = not self.auto_advance
            elif key in (262, 263):  # GLFW right/left arrow
                direction = 1 if key == 262 else -1
                self.cursor = float(np.clip(self.cursor + direction * self.fps / 4, 0, self.frames - 1))
            elif key in (264, 265):  # GLFW down/up arrow
                self.speed = float(np.clip(self.speed * (2 if key == 265 else 0.5), 0.125, 4))

    def verify(self) -> None:
        """Check every reference frame against the model's forward kinematics."""
        for key in self.keys:
            self.select(key)
            position_error = 0.0
            rotation_error = 0.0
            for i in range(self.frames):
                self.apply_frame(i)
                actual = self.data.xpos[self.body_ids]
                position_error = max(
                    position_error, float(np.max(np.linalg.norm(actual - self.clip["body_pos_w"][i], axis=1)))
                )
                actual_q = self.data.xquat[self.body_ids]
                target_q = self.clip["body_quat_w"][i]
                # q and -q encode the same orientation.
                error = np.minimum(
                    np.linalg.norm(actual_q - target_q, axis=1), np.linalg.norm(actual_q + target_q, axis=1)
                )
                rotation_error = max(rotation_error, float(error.max()))
            if position_error > 1e-6 or rotation_error > 1e-6:
                raise ValueError(f"{key}: reference/model mismatch: {position_error=}, {rotation_error=}")
            print(
                f"PASS {key}: all {self.frames} frames; position error {position_error:.3g} m; "
                f"quaternion error {rotation_error:.3g}",
                flush=True,
            )

    def render_previews(self, directory: Path) -> None:
        from PIL import Image  # noqa: PLC0415 -- optional preview dependency

        directory.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(self.model, height=720, width=1280) as renderer:
            for key in self.keys:
                self.select(key)
                self.apply_frame(int(self.frames * 0.45))
                camera = mujoco.MjvCamera()
                self.camera(camera)
                options = mujoco.MjvOption()
                options.geomgroup[3:] = 0  # Hide collision/auxiliary geometry.
                renderer.update_scene(self.data, camera=camera, scene_option=options)
                output = directory / f"g1_reference_{key}.png"
                Image.fromarray(renderer.render()).save(output)
                print(f"Rendered {output}", flush=True)

    def record(self, directory: Path, selected: str) -> None:
        """Render every reference frame directly to MP4 at its recorded frame rate."""
        import imageio.v2 as imageio  # noqa: PLC0415 -- optional recording dependency

        directory.mkdir(parents=True, exist_ok=True)
        keys = self.keys if selected == "all" else [selected]
        encoding = {"codec": "libx264", "quality": 8, "ffmpeg_params": ["-movflags", "+faststart"]}
        with mujoco.Renderer(self.model, height=720, width=1280) as renderer, ExitStack() as stack:
            combined = None
            if len(keys) > 1:
                combined = stack.enter_context(
                    imageio.get_writer(directory / "all_reference_motions.mp4", fps=self.fps * self.speed, **encoding)
                )
            options = mujoco.MjvOption()
            options.geomgroup[3:] = 0
            for key in keys:
                self.select(key)
                camera = mujoco.MjvCamera()
                self.camera(camera)
                output = directory / f"{Path(CLIPS[key][0]).stem}.mp4"
                with imageio.get_writer(output, fps=self.fps * self.speed, **encoding) as writer:
                    for frame in range(self.frames):
                        self.apply_frame(frame)
                        renderer.update_scene(self.data, camera=camera, scene_option=options)
                        pixels = renderer.render()
                        writer.append_data(pixels)
                        if combined is not None:
                            combined.append_data(pixels)
                        if (frame + 1) % 250 == 0:
                            print(f"Recording {key}: {frame + 1}/{self.frames} frames", flush=True)
                print(
                    f"Recorded {output} ({self.frames} frames, {self.frames / self.fps / self.speed:.2f} s)", flush=True
                )
        if len(keys) > 1:
            print(f"Recorded {directory / 'all_reference_motions.mp4'}", flush=True)

    def run(self) -> None:
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.pending_keys.put, show_left_ui=False, show_right_ui=False
        ) as viewer:
            self.camera(viewer.cam)
            viewer.opt.geomgroup[3:] = 0
            previous_time = time.perf_counter()
            while viewer.is_running():
                now = time.perf_counter()
                elapsed = now - previous_time
                previous_time = now
                with viewer.lock():
                    if self.handle_keys():
                        self.camera(viewer.cam)
                        elapsed = 0.0
                    if not self.paused:
                        self.cursor += elapsed * self.fps * self.speed
                    # Hold the final pose for one second before advancing/looping.
                    if self.cursor >= self.frames + self.fps:
                        next_index = (self.keys.index(self.key) + 1) % len(self.keys)
                        self.select(self.keys[next_index] if self.auto_advance else self.key)
                        self.camera(viewer.cam)
                    frame = min(int(self.cursor), self.frames - 1)
                    self.apply_frame(frame)
                # set_texts and sync acquire the viewer mutex themselves.
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f"{CLIPS[self.key][1]}\n{CLIPS[self.key][0]}\n"
                        f"{frame / self.fps:.2f} / {(self.frames - 1) / self.fps:.2f} s | "
                        f"{self.speed:g}x | {'PAUSED' if self.paused else 'PLAYING'} | "
                        f"Auto next: {'ON' if self.auto_advance else 'OFF'}\n"
                        "1/2/3: select | Space: pause | R: restart\n"
                        "Left/Right: seek | Up/Down: speed | A: auto next",
                        "",
                    )
                )
                viewer.sync()
                time.sleep(1 / 120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", choices=["all", *CLIPS], default="all")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--check", action="store_true", help="Validate all frames without opening a window.")
    parser.add_argument("--render-dir", type=Path, help="Render a preview of each motion without opening a window.")
    parser.add_argument("--record-dir", type=Path, help="Record MP4 videos without opening a window.")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    player = ReferencePlayer(args.clip, args.speed)
    if args.check:
        player.verify()
    if args.render_dir:
        player.render_previews(args.render_dir)
    if args.record_dir:
        player.record(args.record_dir, args.clip)
    if not args.check and args.render_dir is None and args.record_dir is None:
        player.run()


if __name__ == "__main__":
    main()
