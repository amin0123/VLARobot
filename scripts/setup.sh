#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v conda >/dev/null || { echo 'Install Miniforge first: https://github.com/conda-forge/miniforge'; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'An NVIDIA RTX GPU/driver is required for Isaac Sim.'; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
mkdir -p outputs/logs
if ! conda run -n vlarobot python --version >/dev/null 2>&1; then
    conda create -y -n vlarobot python=3.12 ffmpeg=7.1 pip
fi
if ! conda run -n vlarobot-sim python --version >/dev/null 2>&1; then
    conda create -y -n vlarobot-sim python=3.12 pip
fi
# The host may source ROS into PYTHONPATH. Conda restores it upon deactivation.
conda env config vars set -n vlarobot PYTHONPATH=
conda env config vars set -n vlarobot-sim PYTHONPATH=
conda run --no-capture-output -n vlarobot python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
conda run --no-capture-output -n vlarobot python -m pip install -e '.[dev]'
conda run --no-capture-output -n vlarobot-sim python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# Minimal packages for our .kit application; no ROS bridge or Isaac Lab.
conda run --no-capture-output -n vlarobot-sim python -m pip install \
    'isaacsim[extscache]==6.1.0.0' isaacsim-app==6.1.0.0 isaacsim-robot==6.1.0.0 \
    isaacsim-sensor==6.1.0.0 isaacsim-core==6.1.0.0 isaacsim-storage==6.1.0.0 \
    --extra-index-url https://pypi.nvidia.com
conda run --no-capture-output -n vlarobot python -m pip check
conda run --no-capture-output -n vlarobot-sim python -m pip check
conda run --no-capture-output -n vlarobot python scripts/check_environment.py
echo 'Activate with: conda activate vlarobot'
