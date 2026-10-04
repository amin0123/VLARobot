# Environment

Inspected 2026-10-04, before installation.

| Item | Observed value |
|---|---|
| OS | Ubuntu 24.04.5 LTS, x86_64 |
| Kernel | Linux 7.0.0-34-generic |
| GPU | NVIDIA GeForce RTX 5070 Ti, Blackwell |
| VRAM | 16,303 MiB total; 1,065 MiB initially used by desktop |
| NVIDIA driver | 595.91.07 |
| Driver CUDA capability | 13.2 (`nvidia-smi`; not an installed CUDA toolkit) |
| System Python | 3.12.3; left untouched |
| Environment manager | Miniforge Conda, `/home/amin24/miniforge3` |
| Existing environments | base, drone, fruitsort; left untouched |
| uv / nvcc | Not found on PATH |
| venv | Available via system Python |
| Git | 2.43.0; repository initially empty |
| RAM / swap | 31 GiB / 8 GiB |
| Free disk | Approximately 650 GiB on project filesystem |

## Selected dependency set (installation/verification in progress)

- Main Conda environment: `vlarobot`, Python 3.12.
- Simulator Conda environment: `vlarobot-sim`, Python 3.12.
- PyTorch: 2.11.0 with CUDA 12.8 wheels; torchvision 0.26.0.
- LeRobot: 0.6.1, upstream tag commit
  `7e241bd630a3719a56157a497ce5d08f244784f1`.
- Isaac Sim: 6.1.0.0, PhysX and RTX RGB camera.
- SmolVLA: official `lerobot/smolvla_base`; revision to be recorded on download.
- FFmpeg: 7.1 from conda-forge.

## Why two environments?

Official PyPI metadata on the inspection date shows incompatible requirements:
Isaac Sim 6.1.0.0 pins NumPy 2.3.1 and packaging 26.0; LeRobot 0.6.1 requires
NumPy <2.3 and packaging <26. Ignoring these constraints would hide a broken
installation. Older releases and a patched dependency specification were
considered. We instead isolate each supported dependency set in its own Conda
environment and pass observations/actions through a local subprocess connection.
This keeps current official APIs and does not change the learned VLA architecture.
No server, cloud account, ROS, detector, or planner is involved in policy control.

Isaac Sim's optional package dependencies may include other simulation libraries;
the project environment and dynamics are Isaac Sim/PhysX. We do not use Isaac Lab.

Official sources checked:

- [Isaac Sim Python installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)
- [LeRobot installation](https://huggingface.co/docs/lerobot/installation)
- [LeRobot 0.6.1 source](https://github.com/huggingface/lerobot/tree/v0.6.1)
- PyPI metadata for `isaacsim-kernel`, `isaacsim-core`, and `lerobot`.

Final installed versions, CUDA verification, and environment snapshots will be
added after the installation passes its checks.

## Host shell isolation

The initial test attempt discovered `/opt/ros/jazzy` in the inherited Python path:
pytest tried loading ROS `launch_testing` and failed on its optional `lark`
dependency. Both dedicated Conda environments now set `PYTHONPATH` to an empty
value (Conda restores the host value when deactivated). The simulator subprocess
also removes inherited `PYTHONHOME`/`LD_LIBRARY_PATH` before its own runtime sets
library paths. No ROS package was changed or added to the project.
