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

## Verified project dependencies

| Component | Version / location |
|---|---|
| Conda environment | `vlarobot`, Python 3.12.14 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| CUDA wheel runtime | 12.8; a real GPU matrix multiplication passed |
| LeRobot | 0.6.1, tag commit `7e241bd630a3719a56157a497ce5d08f244784f1` |
| Transformers / NumPy | 5.5.4 / 2.2.6 |
| TorchCodec / PyAV | 0.11.0+cpu / 15.1.0 |
| FFmpeg | 7.1.1, conda-forge |
| Existing Isaac Sim | `6.0.0-rc.59+release.41464.5f2772bc.gl`, `/home/amin24/isaacsim` |
| Isaac bundled Python | 3.12.13, `/home/amin24/isaacsim/python.sh` |
| SmolVLA checkpoint | `lerobot/smolvla_base`, revision `d9f33c94a60fb382c90dea2164c96845bd955e28` |
| Tokenizer/backbone config | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |

`pip check` passed in `vlarobot`. Loading the complete pretrained SmolVLA weights
on CPU passed (450,046,176 parameters); this is distinct from the later real-image
CUDA forward-pass milestone. Exact runtime tests are tracked in `experiments.md`. Isaac Sim scene rendering,
physics/observation synchronization, and 100 randomized expert episodes have
now passed using the existing standalone runtime.

## Reusing the existing simulator

The first setup attempt overlooked the standalone installation and began a
redundant pip-based Isaac Sim download in a separate `vlarobot-sim` environment.
That installation and its prefetch download were stopped. The task-created
`vlarobot-sim` environment and duplicate Isaac runtime downloads were removed. The corrected setup
uses the existing simulator's bundled Python and creates only the `vlarobot`
environment for project Python libraries. It does not reinstall Isaac Sim or ROS.

A local subprocess bridge isolates Isaac's bundled packages from LeRobot's
packages. This is useful because current pip Isaac Sim 6.1 pins NumPy 2.3.1 and
packaging 26, whereas LeRobot 0.6.1 requires NumPy <2.3 and packaging <26. There
is no need to force these constraints into one interpreter. The bridge carries
observations and actions, and does not change the learned VLA architecture.

`ISAAC_SIM_PATH` overrides the default `~/isaacsim` installation directory.
`VLAROBOT_SIM_PYTHON` can explicitly select an existing runtime launcher.
The project's Kit application uses local extensions with registry downloads
disabled. The existing simulator installation is not upgraded by `setup.sh`.

Official sources checked:

- [Isaac Sim Python installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)
- [LeRobot installation](https://huggingface.co/docs/lerobot/installation)
- [LeRobot 0.6.1 source](https://github.com/huggingface/lerobot/tree/v0.6.1)
- PyPI metadata for `isaacsim-kernel`, `isaacsim-core`, and `lerobot`.

Machine-readable environment evidence is stored in `outputs/environment.json`.

## Host shell isolation

The initial test attempt discovered `/opt/ros/jazzy` in the inherited Python path:
pytest tried loading ROS `launch_testing` and failed on its optional `lark`
dependency. The `vlarobot` Conda environment now sets `PYTHONPATH` to an empty
value (Conda restores the host value when deactivated). The simulator subprocess
also removes inherited `PYTHONHOME`/`LD_LIBRARY_PATH` before its own runtime sets
library paths. No ROS package was changed or added to the project.
