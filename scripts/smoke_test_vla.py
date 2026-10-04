"""Load genuine pretrained weights and test this project's real observation schema."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.common import IMAGE_KEY, write_json
from src.training.model_assets import BACKBONE_REVISION, pin_backbone
from src.training.precision import restore_checkpoint_precision


def main():
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/smolvla_base")
    parser.add_argument("--dataset", default="datasets/splits/train")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", default="outputs/smoke_test_vla.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")
    torch.manual_seed(42)
    ds = LeRobotDataset("local/vlarobot_pick_place_train", root=args.dataset, video_backend="pyav")
    sample = ds[0]
    cfg = SmolVLAConfig.from_pretrained(args.checkpoint)
    cfg.input_features = {
        IMAGE_KEY: PolicyFeature(FeatureType.VISUAL, (3, args.image_size, args.image_size)),
        "observation.state": PolicyFeature(FeatureType.STATE, (14,)),
    }
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    cfg.device, cfg.n_action_steps, cfg.load_vlm_weights = "cuda", 5, False
    pin_backbone(cfg)
    cfg.resize_imgs_with_padding = (args.image_size, args.image_size)
    # All VLM and action expert weights are loaded from SmolVLA's checkpoint.
    # load_vlm_weights=False only avoids redundantly downloading the VLM's weights.
    model = SmolVLAPolicy.from_pretrained(args.checkpoint, config=cfg).to("cuda").eval()
    restored = restore_checkpoint_precision(model, args.checkpoint)
    pre, post = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    batch = pre(
        {
            IMAGE_KEY: sample[IMAGE_KEY],
            "observation.state": sample["observation.state"],
            "task": sample["task"],
        }
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.predict_action_chunk(batch)  # warm-up separate from timed inference
        torch.cuda.synchronize()
        start = time.perf_counter()
        chunk = model.predict_action_chunk(batch)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        action = post(chunk[:, 0])
    assert chunk.shape == (1, cfg.chunk_size, 7)
    assert torch.isfinite(chunk).all() and torch.isfinite(action).all()
    report = {
        "checkpoint": args.checkpoint,
        "backbone_revision": BACKBONE_REVISION,
        "model_image_size": args.image_size,
        "restored_precision_values": restored,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(),
        "input_shapes": {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)},
        "output_shape": list(chunk.shape),
        "action_after_denormalization": action.cpu().tolist(),
        "warm_inference_latency_s": seconds,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
        "allocated_vram_mib": torch.cuda.memory_allocated() / 2**20,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "note": "Base checkpoint smoke test; these actions are not evidence of Panda task competence.",
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
