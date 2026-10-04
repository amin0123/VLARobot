"""Resuming must not mix physical contracts or rewrite inconsistent provenance."""

import copy
import json

import pytest

from src.common import ACTION_NAMES, STATE_NAMES, config, write_json
from src.dataset.collect import features, validate_resume


@pytest.mark.parametrize("change", ["environment", "fingerprint", "state_schema", "sidecar"])
def test_resume_rejects_incompatible_data_without_modification(tmp_path, change):
    env_cfg, cfg = config(), config("dataset")
    gate = {"expert_fingerprint": "unit-test-only"}
    project = {
        "environment": copy.deepcopy(env_cfg),
        "action_names": ACTION_NAMES,
        "state_names": STATE_NAMES,
        "split": cfg["split"],
        "expert_gate": dict(gate),
    }
    info = {
        "fps": env_cfg["control_hz"],
        "robot_type": "franka_panda_cartesian",
        "total_episodes": 1,
        "features": features(env_cfg["image_size"]),
    }
    sidecar = [{"episode_index": 0, "seed": 123}]

    def save():
        write_json(tmp_path / "meta/project.json", project)
        write_json(tmp_path / "meta/info.json", info)
        (tmp_path / "meta/collection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sidecar))

    save()
    assert validate_resume(tmp_path, env_cfg, cfg, gate) == sidecar
    if change == "environment":
        project["environment"]["control_hz"] = 20
    elif change == "fingerprint":
        project["expert_gate"]["expert_fingerprint"] = "another-controller"
    elif change == "state_schema":
        info["features"]["observation.state"]["shape"] = (15,)
    else:
        sidecar.append({"episode_index": 1, "seed": 123})
    save()
    before = {p.name: p.read_bytes() for p in (tmp_path / "meta").iterdir()}
    with pytest.raises(ValueError):
        validate_resume(tmp_path, env_cfg, cfg, gate)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "meta").iterdir()}
