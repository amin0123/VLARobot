"""Small synchronous bridge between isolated LeRobot and Isaac Sim interpreters."""

import os
import subprocess
import tempfile
import time
from multiprocessing.connection import Listener
from pathlib import Path

from src.common import ROOT
from src.simulation.runtime import simulator_python


class SimulationClient:
    def __init__(self, gui=False, cfg="environment", log_path=None, startup_timeout=600):
        command = [str(simulator_python())]
        self.directory = tempfile.TemporaryDirectory(prefix="vlarobot-")
        self.socket_path = str(Path(self.directory.name) / "sim.sock")
        self.listener = Listener(self.socket_path, family="AF_UNIX")
        self.listener._listener._socket.settimeout(1.0)
        self.conn = None
        self.log_path = Path(log_path or ROOT / "outputs/logs/simulation.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        command += ["-u", "-m", "src.simulation.worker", "--socket", self.socket_path, "--config", str(cfg)]
        if gui:
            command.append("--gui")
        child_env = os.environ.copy()
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONEXE",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "CONDA_PREFIX",
        ):
            child_env.pop(name, None)
        child_env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=child_env,
            start_new_session=True,
        )
        deadline = time.monotonic() + startup_timeout
        try:
            while self.conn is None:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Isaac Sim exited ({self.process.returncode}). See {self.log_path}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Isaac Sim startup timed out. See {self.log_path}")
                try:
                    self.conn = self.listener.accept()
                except TimeoutError:
                    continue
            self.ready = self._receive(timeout=max(1, deadline - time.monotonic()))
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout=120):
        if not self.conn.poll(timeout):
            raise TimeoutError(f"Simulation response timed out. See {self.log_path}")
        response = self.conn.recv()
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def call(self, cmd, **kwargs):
        self.conn.send({"cmd": cmd, **kwargs})
        return self._receive()

    def reset(self, seed, task, condition="train", render=True, episode=0):
        if hasattr(task, "to_dict"):
            task = task.to_dict()
        return self.call(
            "reset", seed=int(seed), task=task, condition=condition, render=render, episode=int(episode)
        )

    def step(self, action, render=True):
        return self.call("step", action=action, render=render)

    def close(self):
        import signal

        process = getattr(self, "process", None)
        if self.conn is not None:
            try:
                self.conn.send({"cmd": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.conn.close()
            self.conn = None
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        self.listener.close()
        self.log.close()
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
