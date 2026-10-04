"""Select CUDA BF16 autocast, then delegate entirely to the official trainer.

LeRobot 0.6.1 selects autocast using torch.get_autocast_dtype for SmolVLA,
whose config has use_amp but no dtype field. CUDA's FP16 default would create
a GradScaler that cannot unscale this pretrained checkpoint's BF16 gradients.
"""

import torch


def main():
    torch.set_autocast_dtype("cuda", torch.bfloat16)
    from lerobot.scripts.lerobot_train import main as lerobot_main

    print(
        "VLA trainer: CUDA autocast dtype is bfloat16; using the official LeRobot training loop.", flush=True
    )
    lerobot_main()


if __name__ == "__main__":
    main()
