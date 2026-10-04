"""Preserve saved FP32 weights and select optimizer parameter precision."""

from pathlib import Path

import torch
from safetensors import safe_open


def restore_checkpoint_precision(policy, checkpoint):
    """Undo FP32-to-low-precision casts after the official strict weight load.

        SmolVLA's nested Transformers config can instantiate expert layers in BF16 even
        when fine-tuning saved FP32 parameters. Re-read only parameters whose storage
    dtype was narrowed, preserving saved values and existing Parameter objects. Keep
    the upstream constructor's choice when it is already at least FP32 precision.
    """
    path = Path(checkpoint) / "model.safetensors"
    if not path.is_file():
        return 0
    restored = 0
    with safe_open(path, framework="pt", device="cpu") as weights, torch.no_grad():
        names = set(weights.keys())
        for name, parameter in policy.named_parameters():
            if name not in names:
                continue  # The official loader has already checked keys/shapes.
            saved_dtype = weights.get_slice(name).get_dtype()
            if saved_dtype == "F32" and parameter.dtype in (torch.float16, torch.bfloat16):
                parameter.data = weights.get_tensor(name).to(device=parameter.device, dtype=torch.float32)
                restored += parameter.numel()
    return restored


def promote_trainable_parameters(policy):
    """Keep optimizer updates in FP32; frozen weights retain their existing dtype."""
    promoted = 0
    for parameter in policy.parameters():
        if parameter.requires_grad and parameter.is_floating_point() and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
            promoted += parameter.numel()
    return promoted
