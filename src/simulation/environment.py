"""Isaac Sim environment; import this only AFTER starting SimulationApp.

Public observations are intentionally separate from privileged evaluator/teacher
telemetry. All manipulation between resets uses PhysX contacts and joint drives.
"""

import numpy as np

from src.common import IMAGE_KEY, RGB, config
from src.control.ik_controller import CartesianServo
from src.control.scripted_policy import ScriptedExpert
from src.evaluation.metrics import contained
from src.simulation.camera import RGBCamera
from src.simulation.objects import sample_layout
from src.simulation.robot import configure_drives, create_robot


class PickPlaceEnvironment:
    def __init__(self, app, cfg=None, headless=True):
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from pxr import UsdLux, UsdPhysics, UsdShade

        self.app, self.cfg = app, cfg or config()
        c = self.cfg
        if c["physics_hz"] % c["control_hz"]:
            raise ValueError("physics_hz must be divisible by control_hz")
        self.substeps = c["physics_hz"] // c["control_hz"]
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1 / c["physics_hz"],
            rendering_dt=1 / c["control_hz"],
            backend="numpy",
            device="cpu",
        )
        self.world.get_physics_context().enable_gpu_dynamics(False)
        self.world.get_physics_context().set_broadphase_type("MBP")
        self.stage = omni.usd.get_context().get_stage()
        self.material = PhysicsMaterial(
            "/World/Materials/Grip", static_friction=1.2, dynamic_friction=1.0, restitution=0.0
        )
        self.world.scene.add(
            FixedCuboid(
                "/World/Table",
                name="table",
                position=np.array([0.4, 0.0, -0.035]),
                scale=np.array([1.05, 1.05, 0.07]),
                color=np.array([0.42, 0.36, 0.28]),
                size=1.0,
                physics_material=self.material,
            )
        )
        self.world.scene.add_default_ground_plane(z_position=-0.075)
        self.robot, self.asset_path = create_robot(self.world, c)
        self.cubes = {}
        for color, position in sample_layout(c, 0).items():
            self.cubes[color] = self.world.scene.add(
                DynamicCuboid(
                    f"/World/Cubes/{color}",
                    name=f"cube_{color}",
                    position=position,
                    size=c["cube_size"],
                    mass=c["cube_mass"],
                    color=np.array(RGB[color]),
                    physics_material=self.material,
                )
            )
        # A container consists of five collision boxes, with an open top.
        inner, thick, height, floor = (
            c[k] for k in ("bin_inner_width", "bin_wall_thickness", "bin_wall_height", "bin_floor_thickness")
        )
        for color, xy in c["container_centers"].items():
            parts = [("floor", [*xy, floor / 2], [inner + 2 * thick, inner + 2 * thick, floor])]
            for axis in (0, 1):
                for sign in (-1, 1):
                    position = [*xy, floor + height / 2]
                    position[axis] += sign * (inner + thick) / 2
                    scale = [inner + 2 * thick, inner + 2 * thick, height]
                    scale[axis] = thick
                    parts.append((f"wall_{axis}_{'p' if sign > 0 else 'm'}", position, scale))
            for part, position, scale in parts:
                self.world.scene.add(
                    FixedCuboid(
                        f"/World/Containers/{color}/{part}",
                        name=f"{color}_{part}",
                        position=np.array(position),
                        scale=np.array(scale),
                        size=1.0,
                        color=np.array(RGB[color]) * 0.8 + 0.12,
                        physics_material=self.material,
                    )
                )
        self.light = UsdLux.DomeLight.Define(self.stage, "/World/Light")
        self.light.CreateIntensityAttr(c["dome_intensity"])
        self.light.CreateColorAttr((1.0, 1.0, 1.0))
        # Bind friction to the existing finger colliders, without changing geometry.
        for prim in self.stage.Traverse():
            if "finger" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    self.material.material, materialPurpose="physics"
                )
        self.camera = RGBCamera(c)
        self.world.reset()
        configure_drives(self.robot)
        self.servo = CartesianServo(self.robot, c)
        self.ui_label = None
        if not headless:
            import omni.ui as ui
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport:
                viewport.camera_path = "/World/Camera"
            self.ui_window = ui.Window("VLA episode", width=600, height=130)
            with self.ui_window.frame:
                self.ui_label = ui.Label("Initializing…", word_wrap=True)
        self.task = None
        self.step_count = 0
        self.success_count = 0
        self.success = False
        self.side = c["cube_size"]
        self.initial_configuration = None

    def reset(self, seed, task, condition="train", render=True):
        self.task, self.seed, self.condition = task, int(seed), condition
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 512]))
        scale = float(rng.uniform(0.9, 1.1)) if condition == "scale" else 1.0
        new_side = self.cfg["cube_size"] * scale
        if not np.isclose(self.side, new_side, rtol=0, atol=1e-12):
            # A changed collision scale invalidates active PhysX tensor views.
            # Author it while stopped; World.reset reinitializes scene handles.
            self.world.stop()
            for cube in self.cubes.values():
                cube.set_local_scale(np.full(3, scale))
        self.world.reset()
        configure_drives(self.robot)
        self.side = self.cfg["cube_size"] * scale
        for color, position in sample_layout(self.cfg, seed, scale).items():
            cube = self.cubes[color]
            cube.set_local_scale(np.full(3, scale))
            cube.set_world_pose(position, np.array([1.0, 0.0, 0.0, 0.0]))
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        q = np.array([*self.cfg["home_joints"], 0.04, 0.04])
        self.robot.set_joint_positions(q)
        self.robot.set_joint_velocities(np.zeros(9))
        from isaacsim.core.utils.types import ArticulationAction

        self.robot.apply_action(ArticulationAction(joint_positions=q))
        self.light.GetIntensityAttr().Set(
            self.cfg["dome_intensity"] * (float(rng.uniform(0.65, 1.35)) if condition == "lighting" else 1)
        )
        self.camera.reset(rng, condition)
        for _ in range(60):
            self.world.step(render=False)
        if render:
            self._render()
        self.step_count, self.success_count, self.success = 0, 0, False
        self.expert = ScriptedExpert(self.cfg)
        self.initial_configuration = self.privileged()
        self._update_ui("ready")
        return self.observe(render=False, include_image=render), self.info()

    def _render(self):
        import omni.replicator.core as rep

        # Explicit synchronous capture is required by this build's sensor pipeline.
        # A zero time delta captures the current physics state without advancing it.
        self.world.render()
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)

    def observe(self, render=False, include_image=True):
        if render:
            self._render()
        tcp, angles = self.servo.pose()
        q = self.robot.get_joint_positions()
        state = np.array([*tcp, *angles, q[7] + q[8], *q[:7]], dtype=np.float32)
        obs = {"observation.state": state}
        if include_image:
            obs[IMAGE_KEY] = self.camera.read()
        return obs

    def privileged(self):
        tcp, _ = self.servo.pose()
        objects, orientations, velocities = {}, {}, {}
        for color, cube in self.cubes.items():
            p, q = cube.get_world_pose()
            objects[color], orientations[color] = p.tolist(), q.tolist()
            velocities[color] = cube.get_linear_velocity().tolist()
        return {
            "tcp": tcp.tolist(),
            "objects": objects,
            "orientations": orientations,
            "velocities": velocities,
            "gripper_width": float(sum(self.robot.get_joint_positions()[7:9])),
            "step": self.step_count,
            "cube_size": self.side,
        }

    def _check_success(self):
        p, q = self.cubes[self.task["source"]].get_world_pose()
        speed = np.linalg.norm(self.cubes[self.task["source"]].get_linear_velocity())
        tcp, _ = self.servo.pose()
        released = sum(self.robot.get_joint_positions()[7:9]) > 0.065 and np.linalg.norm(tcp - p) > 0.085
        inside = contained(p, q, self.side, self.cfg["container_centers"][self.task["destination"]], self.cfg)
        stable = inside and released and speed < 0.02
        self.success_count = self.success_count + 1 if stable else 0
        self.success = self.success_count >= self.cfg["success_hold_steps"]

    def step(self, action, render=True):
        self.servo.apply(action)
        for _ in range(self.substeps):
            self.world.step(render=False)
        self.step_count += 1
        if render:
            self._render()
        self._check_success()
        self._update_ui("success" if self.success else "running")
        return self.observe(include_image=render), self.info()

    def expert_action(self):
        return self.expert.action(self.privileged(), self.task)

    def info(self):
        return {
            "success": self.success,
            "step": self.step_count,
            "terminated": self.success,
            "truncated": self.step_count >= self.cfg["max_steps"],
            "sim_seconds": self.step_count / self.cfg["control_hz"],
            "ik_success": self.servo.last_ik_success,
            "action_clipped": self.servo.last_action_clipped,
            "seed": getattr(self, "seed", None),
            "initial_configuration": self.initial_configuration,
            "asset_path": self.asset_path,
        }

    def _update_ui(self, status):
        if self.ui_label is not None and self.task:
            self.ui_label.text = (
                f"{self.task['instruction']}\nSeed {self.seed} | Step {self.step_count} | {status}"
            )
