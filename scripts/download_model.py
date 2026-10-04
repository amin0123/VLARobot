"""Fetch the exact official base checkpoint used by this project."""

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.model_assets import BACKBONE, BACKBONE_REVISION

REPO = "lerobot/smolvla_base"
REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"


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
