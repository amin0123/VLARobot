"""A pose servo, not a semantic planner. It never reads task or object state."""

import numpy as np
from scipy.spatial.transform import Rotation


class CartesianServo:
    def __init__(self, robot, cfg):
        from isaacsim.robot_motion.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
            interface_config_loader,
        )

        self.robot, self.cfg = robot, cfg
        lula_cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self.lula = LulaKinematicsSolver(**lula_cfg)
        self.solver = ArticulationKinematicsSolver(robot, self.lula, "panda_hand")
        self.last_ik_success = True
        self.last_action_clipped = False

    def pose(self):
        hand, rotation = self.solver.compute_end_effector_pose()
        tcp = hand + rotation @ np.array([0.0, 0.0, self.cfg["tcp_offset"]])
        angles = Rotation.from_matrix(rotation).as_euler("xyz")
        # Canonicalize near pi to keep the state continuous around the downward pose.
        if angles[0] < 0:
            angles[0] += 2 * np.pi
        return tcp, angles

    def apply(self, action):
        from isaacsim.core.utils.types import ArticulationAction

        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise ValueError(f"Expected seven finite Cartesian action values, got {raw}.")
        xyz, angles = self.pose()
        target = np.clip(raw[:3], self.cfg["workspace_low"], self.cfg["workspace_high"])
        delta = target - xyz
        distance = np.linalg.norm(delta)
        if distance > self.cfg["max_translation_step"]:
            target = xyz + delta * self.cfg["max_translation_step"] / distance
        target_angles = np.clip(raw[3:6], [np.pi - 0.2, -0.2, -0.4], [np.pi + 0.2, 0.2, 0.4])
        target_angles = angles + np.clip(
            target_angles - angles, -self.cfg["max_rotation_step"], self.cfg["max_rotation_step"]
        )
        rotation = Rotation.from_euler("xyz", target_angles)
        quaternion = rotation.as_quat()[[3, 0, 1, 2]]
        hand_target = target - rotation.apply([0.0, 0.0, self.cfg["tcp_offset"]])
        joint_action, success = self.solver.compute_inverse_kinematics(
            hand_target,
            quaternion,
            position_tolerance=0.001,
            orientation_tolerance=0.02,
        )
        self.last_ik_success = bool(success)
        self.last_action_clipped = bool(np.linalg.norm(target - raw[:3]) > 1e-6)
        if success:
            self.robot.apply_action(joint_action)
        opening = float(np.clip(raw[6], 0.0, 1.0)) * 0.04
        self.robot.apply_action(
            ArticulationAction(joint_positions=np.array([opening, opening]), joint_indices=np.array([7, 8]))
        )
