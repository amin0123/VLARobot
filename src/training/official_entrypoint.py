"""Select CUDA BF16 autocast, then delegate entirely to the official trainer.

LeRobot 0.6.1 selects autocast using torch.get_autocast_dtype for SmolVLA,
whose config has use_amp but no dtype field. CUDA's FP16 default would create
a GradScaler that cannot unscale this pretrained checkpoint's BF16 gradients.
"""

import os

import torch

from src.training.model_assets import pin_backbone, pinned_processor_overrides
from src.training.precision import promote_trainable_parameters, restore_checkpoint_precision


def main():
    torch.set_autocast_dtype("cuda", torch.bfloat16)
    from lerobot.scripts import lerobot_train

    factory = lerobot_train.make_policy
    processor_factory = lerobot_train.make_pre_post_processors

    def make_policy_with_precision(cfg, **kwargs):
        pin_backbone(cfg)
        policy = factory(cfg=cfg, **kwargs)
        if policy.config.pretrained_path:
            restored = restore_checkpoint_precision(policy, policy.config.pretrained_path)
            print(f"Restored saved storage precision for {restored:,} parameter values.", flush=True)
        if os.environ.get("VLAROBOT_TRAINABLE_DTYPE", "checkpoint") == "float32":
            promoted = promote_trainable_parameters(policy)
            print(f"Promoted {promoted:,} trainable parameter values to FP32 storage.", flush=True)
        return policy

    def make_pinned_processors(policy_cfg, **kwargs):
        kwargs["preprocessor_overrides"] = pinned_processor_overrides(
            policy_cfg, kwargs.get("preprocessor_overrides")
        )
        return processor_factory(policy_cfg=policy_cfg, **kwargs)

    # This process-local construction hook runs before upstream creates the
    # optimizer. Batches, updates, saves, and scheduling stay upstream.
    lerobot_train.make_policy = make_policy_with_precision
    lerobot_train.make_pre_post_processors = make_pinned_processors

    print(
        "VLA trainer: CUDA autocast dtype is bfloat16; using the official LeRobot training loop.", flush=True
    )
    lerobot_train.main()


if __name__ == "__main__":
    main()
