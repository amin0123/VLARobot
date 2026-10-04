"""Isaac-only process. IPC is local in a private temporary directory."""

import argparse
import os
import sys
import traceback
from multiprocessing.connection import Client

from src.common import ROOT, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--config", default="environment")
    args = parser.parse_args()
    # Reuse the user's existing standalone runtime and its bundled interpreter.
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    isaac_path = os.environ["ISAAC_PATH"]
    # Do not pass this module's arguments into Kit's own argument parser.
    sys.argv = [sys.argv[0]]
    app = SimulationApp(
        {
            "headless": not args.gui,
            "width": 640,
            "height": 640,
            "window_width": 1000,
            "window_height": 800,
            "renderer": "RaytracedLighting",
            "multi_gpu": False,
            "disable_viewport_updates": not args.gui,
            "limit_cpu_threads": 8,
            "extra_args": [
                "--ext-folder",
                isaac_path + "/extscache",
                "--ext-folder",
                isaac_path + "/extsDeprecated",
                "--/log/level=warning",
                "--/log/outputStreamLevel=warning",
                "--/exts/omni.kit.renderer.core/present/enabled=false",
            ],
        },
        experience=str(ROOT / "configs/vlarobot.kit"),
    )
    conn = Client(args.socket, family="AF_UNIX")
    try:
        from src.simulation.environment import PickPlaceEnvironment

        env = PickPlaceEnvironment(app, config(args.config), headless=not args.gui)
        conn.send({"ok": True, "result": {"ready": True, "asset_path": env.asset_path}})
        episode = 0
        while app.is_running():
            msg = conn.recv()
            try:
                cmd = msg.pop("cmd")
                if cmd == "close":
                    conn.send({"ok": True, "result": None})
                    break
                if cmd == "reset":
                    episode = msg.pop("episode", 0)
                    result = env.reset(**msg)
                elif cmd == "step":
                    result = env.step(**msg)
                elif cmd == "observe":
                    result = env.observe(**msg)
                elif cmd == "privileged":
                    result = env.privileged()
                elif cmd == "expert_action":
                    result = env.expert_action()
                elif cmd == "status":
                    env._update_ui(msg["text"])
                    env.world.render()
                    result = None
                else:
                    raise ValueError(f"Unknown worker command {cmd}")
                if env.ui_label is not None and cmd in ("reset", "step", "status"):
                    env.ui_label.text = f"Episode {episode} | {env.ui_label.text}"
                conn.send({"ok": True, "result": result})
            except Exception:
                conn.send({"ok": False, "error": traceback.format_exc()})
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        conn.send({"ok": False, "error": traceback.format_exc()})
        raise
    finally:
        conn.close()
        app.close()


if __name__ == "__main__":
    main()
