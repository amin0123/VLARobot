"""Strict RGB + language + proprioception boundary around official SmolVLA."""

import time

import numpy as np
import torch

from src.common import IMAGE_KEY


def policy_observation(observation, instruction):
    """Allow-list the only two sensor channels, never pass an env info dictionary."""
    rgb = observation[IMAGE_KEY]
    state = observation["observation.state"]
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected HWC uint8 RGB")
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Expected 14 finite proprioceptive values")
    return {
        IMAGE_KEY: torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
        "observation.state": torch.from_numpy(state.copy()).float(),
        "task": instruction,
    }


class VLAPolicy:
    def __init__(self, checkpoint, device="cuda", n_action_steps=5):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        cfg = SmolVLAConfig.from_pretrained(checkpoint)
        cfg.device, cfg.n_action_steps, cfg.load_vlm_weights = device, n_action_steps, False
        cfg.push_to_hub = False
        if not 1 <= n_action_steps <= cfg.chunk_size:
            raise ValueError("n_action_steps must be between 1 and the checkpoint chunk_size")
        if cfg.input_features["observation.state"].shape != (14,) or cfg.output_features["action"].shape != (
            7,
        ):
            raise ValueError(
                "Use a Panda fine-tuned checkpoint (state=14, action=7), not the original SO100 configuration."
            )
        self.model = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.device, self.last_latency_s = device, 0.0
        self.calls = 0

    def reset(self, seed=None):
        if seed is not None:
            # SmolVLA samples Gaussian flow noise. Pair this seed across evaluation
            # conditions as well as pairing the simulator's scene seed.
            torch.manual_seed(int(seed))
        self.model.reset()
        self.pre.reset()
        self.post.reset()
        self.calls = 0

    @torch.inference_mode()
    def act(self, observation, instruction):
        batch = self.pre(policy_observation(observation, instruction))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        # Official synchronous action queue: predict a chunk when empty, execute
        # n_action_steps of its prefix, then condition a fresh chunk on new RGB.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            action = self.model.select_action(batch)
        action = self.post(action)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.last_latency_s = time.perf_counter() - start
        self.calls += 1
        result = action.squeeze(0).cpu().float().numpy()
        if result.shape != (7,) or not np.isfinite(result).all():
            raise RuntimeError(f"Non-finite or wrong-sized policy action: {result}")
        return result
