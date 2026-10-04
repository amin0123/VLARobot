"""Privileged teacher used only for demonstrations and expert validation."""

import numpy as np


class ScriptedExpert:
    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = 0
        self.ticks = 0
        self.settle_ticks = 0
        self.grasp_position = None

    @property
    def name(self):
        return ("approach", "descend", "close", "lift", "transfer", "lower", "release", "retreat", "settle")[
            min(self.phase, 8)
        ]

    def action(self, privileged, task):
        """Return the same 7D action the VLA will learn, before environment.step."""
        cube = np.asarray(privileged["objects"][task["source"]])
        tcp = np.asarray(privileged["tcp"])
        destination = np.asarray(self.cfg["container_centers"][task["destination"]])
        if self.grasp_position is None:
            self.grasp_position = cube.copy()
        grasp = self.grasp_position
        floor = self.cfg["table_height"] + self.cfg["bin_floor_thickness"]
        targets = [
            np.array([*grasp[:2], 0.22]),
            np.array([*grasp[:2], grasp[2]]),
            np.array([*grasp[:2], grasp[2]]),
            np.array([*grasp[:2], 0.24]),
            np.array([*destination, 0.24]),
            np.array([*destination, floor + self.cfg["cube_size"] / 2 + 0.006]),
            np.array([*destination, floor + self.cfg["cube_size"] / 2 + 0.006]),
            np.array([*destination, 0.25]),
            np.array([*destination, 0.25]),
        ]
        phase = min(self.phase, 8)
        goal = targets[phase]
        gripper = 0.0 if 2 <= phase <= 5 else 1.0
        action = np.array([*goal, np.pi, 0, 0, gripper], dtype=np.float32)
        self.ticks += 1
        close = np.linalg.norm(tcp - goal) < (0.006 if phase in (1, 5) else 0.012)
        # Contact means closed fingers cannot reach zero; phase timing must not wait for zero.
        minimum = {0: 4, 1: 3, 2: 8, 3: 4, 4: 4, 5: 4, 6: 6, 7: 5, 8: 10}[phase]
        ready = self.ticks >= minimum and (close or phase in (2, 6, 8))
        if ready and phase < 8:
            self.phase += 1
            self.ticks = 0
        return action
