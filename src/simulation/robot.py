"""Franka setup. Joint teleports are restricted to episode reset."""

import os

import numpy as np


def create_robot(world, cfg):
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path

    asset = os.environ.get("VLAROBOT_FRANKA_USD")
    if not asset:
        root = get_assets_root_path()
        if not root:
            raise RuntimeError(
                "Isaac asset server unavailable. Set VLAROBOT_FRANKA_USD to a local Franka USD."
            )
        asset = root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(asset, "/World/Franka")
    robot = world.scene.add(SingleArticulation(prim_path="/World/Franka", name="franka"))
    robot.set_joints_default_state(positions=np.array([*cfg["home_joints"], 0.04, 0.04]))
    return robot, asset


def configure_drives(robot):
    # Panda arm and parallel jaws remain physically actuated; never weld an object.
    robot.get_articulation_controller().set_gains(
        kps=np.array([4000.0] * 7 + [1200.0, 1200.0]),
        kds=np.array([200.0] * 7 + [80.0, 80.0]),
    )
    robot.set_solver_position_iteration_count(16)
    robot.set_solver_velocity_iteration_count(4)
