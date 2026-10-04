"""Resolve the exact backbone configuration/tokenizer without following Hub main."""

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def pin_backbone(config):
    # Saved checkpoints may contain the absolute cache path from another machine.
    if config.vlm_model_name != BACKBONE and Path(config.vlm_model_name).name != BACKBONE_REVISION:
        raise ValueError(f"This project expects {BACKBONE}; review the model contract before changing it.")
    try:
        path = Path(
            snapshot_download(
                BACKBONE,
                revision=BACKBONE_REVISION,
                allow_patterns=["*.json", "*.txt"],
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            "Run python scripts/download_model.py to fetch the pinned backbone assets."
        ) from exc
    required = (
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    if any(not (path / name).is_file() for name in required):
        raise RuntimeError("Pinned backbone assets are incomplete; run python scripts/download_model.py.")
    config.vlm_model_name = str(path)
    return str(path)


def pinned_processor_overrides(config, overrides=None):
    result = dict(overrides or {})
    result["tokenizer_processor"] = {
        **result.get("tokenizer_processor", {}),
        "tokenizer_name": pin_backbone(config),
    }
    return result
