import numpy as np
import pytest

from src.common import IMAGE_KEY
from src.inference.policy import policy_observation


def test_ground_truth_cannot_enter_policy_observation():
    observation = {
        IMAGE_KEY: np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.state": np.zeros(14, dtype=np.float32),
        "object_poses": {"red": [0.4, 0.2, 0.02]},
        "destination": [0.66, 0.09],
        "success": True,
    }
    result = policy_observation(observation, "Put the red cube in the blue box.")
    assert set(result) == {IMAGE_KEY, "observation.state", "task"}
    assert result[IMAGE_KEY].shape == (3, 256, 256)
    assert result["observation.state"].shape == (14,)


def test_rejects_incompatible_sensor_contract():
    with pytest.raises(ValueError, match="14 finite"):
        policy_observation(
            {
                IMAGE_KEY: np.zeros((256, 256, 3), dtype=np.uint8),
                "observation.state": np.zeros(6, dtype=np.float32),
            },
            "test",
        )
