"""Record installed packages and verify a real CUDA computation."""

import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import write_json


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not start training.")
    a = torch.randn(256, 256, device="cuda")
    result = a @ a.T
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    report = {
        "os": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "gpu": torch.cuda.get_device_name(),
        "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_architectures": torch.cuda.get_arch_list(),
        "compute_capability": torch.cuda.get_device_capability(),
        "cuda_matmul_passed": True,
        "packages": {
            name: md.version(name)
            for name in ["torch", "torchvision", "torchcodec", "lerobot", "transformers", "numpy", "av"]
        },
        "nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True
        ),
    }
    write_json(ROOT / "outputs/environment.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
