"""Fetch the exact official base checkpoint used by this project."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "lerobot/smolvla_base"
REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"
BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def main():
    root = Path(__file__).resolve().parents[1] / "checkpoints/smolvla_base"
    snapshot_download(
        REPO, revision=REVISION, local_dir=root, allow_patterns=["*.json", "*.safetensors", "README.md"]
    )
    # Configuration/tokenizer only: the SmolVLA checkpoint contains the actual
    # pretrained backbone weights, so a second model weight download is redundant.
    snapshot_download(BACKBONE, revision=BACKBONE_REVISION, allow_patterns=["*.json", "*.txt"])
    (root / "upstream_revision.json").write_text(
        json.dumps(
            {
                "repo_id": REPO,
                "revision": REVISION,
                "backbone": BACKBONE,
                "backbone_revision": BACKBONE_REVISION,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Downloaded {REPO}@{REVISION} into {root}")


if __name__ == "__main__":
    main()
