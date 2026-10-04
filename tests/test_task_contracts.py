import numpy as np
import pytest

from src.common import config
from src.evaluation.metrics import contained, wilson_interval
from src.simulation.objects import sample_layout
from src.tasks import (
    HELD_OUT_PAIRS,
    TEST_TEMPLATES,
    TRAIN_PAIRS,
    TRAIN_TEMPLATES,
    sample_task,
    task_from_instruction,
)


def test_splits_do_not_leak_compositions_or_phrases():
    assert not set(HELD_OUT_PAIRS) & set(TRAIN_PAIRS)
    assert not set(TRAIN_TEMPLATES) & set(TEST_TEMPLATES)
    for i in range(100):
        task = sample_task(i, i)
        assert (task.source, task.destination) in TRAIN_PAIRS
        task = sample_task(i, i, "composition")
        assert (task.source, task.destination) in HELD_OUT_PAIRS


def test_reset_geometry_many_seeds():
    cfg = config()
    for seed in range(1000):
        positions = np.asarray(list(sample_layout(cfg, seed).values()))
        assert positions.shape == (4, 3)
        for i in range(4):
            for j in range(i):
                assert np.linalg.norm(positions[i, :2] - positions[j, :2]) >= cfg["minimum_cube_distance"]
        for xy in cfg["container_centers"].values():
            assert np.all(np.linalg.norm(positions[:, :2] - xy, axis=1) > cfg["bin_inner_width"] / 2)
        assert np.all(positions >= np.array(cfg["workspace_low"]))
        assert np.all(positions <= np.array(cfg["workspace_high"]))


def test_success_requires_full_cube_inside_and_resting():
    cfg = config()
    xy = np.asarray(cfg["container_centers"]["blue"])
    p = np.array([*xy, cfg["bin_floor_thickness"] + cfg["cube_size"] / 2])
    assert contained(p, [1, 0, 0, 0], 0.04, xy, cfg)
    assert not contained(p + [0.045, 0, 0], [1, 0, 0, 0], 0.04, xy, cfg)
    assert not contained(p + [0, 0, 0.1], [1, 0, 0, 0], 0.04, xy, cfg)


def test_intervals_handle_extreme_results():
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(100, 100)
    assert 0.96 < low < 0.97 and high == pytest.approx(1)
    low, high = wilson_interval(0, 100)
    assert low == pytest.approx(0) and 0.03 < high < 0.04


def test_metric_parser_preserves_model_language():
    instruction = "Put the green cube in the red box."
    task = task_from_instruction(instruction)
    assert task.instruction == instruction
    assert (task.source, task.destination) == ("green", "red")
    with pytest.raises(ValueError):
        task_from_instruction("Put that one there.")
