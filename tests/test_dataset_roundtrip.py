"""Tiny synthetic API fixture, never used as a demonstration or experiment result."""

import numpy as np

from src.common import IMAGE_KEY
from src.dataset.collect import features


def test_lerobot_writer_roundtrip_preserves_alignment(tmp_path):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    root = tmp_path / "api_fixture"
    writer = LeRobotDataset.create(
        repo_id="local/api_fixture",
        root=root,
        fps=10,
        features=features(32),
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18),
        image_writer_threads=2,
        encoder_threads=1,
    )
    for index in range(5):
        writer.add_frame(
            {
                IMAGE_KEY: np.full((32, 32, 3), index * 40, dtype=np.uint8),
                "observation.state": np.full(14, index, dtype=np.float32),
                "action": np.full(7, index + 0.25, dtype=np.float32),
                "task": "API fixture only",
                "next.done": np.array([index == 4]),
                "next.success": np.array([index == 4]),
            }
        )
    writer.save_episode()
    writer.finalize()
    reader = LeRobotDataset("local/api_fixture", root=root, video_backend="pyav")
    assert reader.num_episodes == 1 and reader.num_frames == 5 and reader.fps == 10
    for index in range(5):
        frame = reader[index]
        np.testing.assert_allclose(frame["observation.state"], index)
        np.testing.assert_allclose(frame["action"], index + 0.25)
        assert frame[IMAGE_KEY].shape == (3, 32, 32)
        assert frame["task"] == "API fixture only"
        assert bool(frame["next.success"].item()) == (index == 4)

    # Exercise the real append API as well: old frames and episode indices must survive.
    writer = LeRobotDataset.resume(
        repo_id="local/api_fixture",
        root=root,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18),
        image_writer_threads=2,
        encoder_threads=1,
    )
    for index in range(5, 8):
        writer.add_frame(
            {
                IMAGE_KEY: np.full((32, 32, 3), index * 30, dtype=np.uint8),
                "observation.state": np.full(14, index, dtype=np.float32),
                "action": np.full(7, index + 0.25, dtype=np.float32),
                "task": "Resumed API fixture only",
                "next.done": np.array([index == 7]),
                "next.success": np.array([index == 7]),
            }
        )
    writer.save_episode()
    writer.finalize()
    reader = LeRobotDataset("local/api_fixture", root=root, video_backend="pyav")
    assert reader.num_episodes == 2 and reader.num_frames == 8
    for index in range(8):
        frame = reader[index]
        np.testing.assert_allclose(frame["observation.state"], index)
        np.testing.assert_allclose(frame["action"], index + 0.25)
        assert int(frame["episode_index"]) == int(index >= 5)
        expected_task = "API fixture only" if index < 5 else "Resumed API fixture only"
        assert frame["task"] == expected_task
