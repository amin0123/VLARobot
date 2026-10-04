# VLA Robot Lab

An educational language-controlled Panda manipulation project using **Isaac Sim,
Hugging Face LeRobot, and pretrained SmolVLA**. Follow the pipeline from physical
expert demonstrations to a LeRobot dataset, fine-tuning, closed-loop robot control,
and honest generalization evaluation.

**Executed end to end:** the existing Isaac Sim scene and synchronized RGB capture
pass runtime checks; the scripted expert achieved **100/100** randomized successes.
The real dataset contains **50 demonstrations / 3,914 frames**, with 45/5
train/validation episodes and reviewed replays. Pretrained CUDA inference, sanity
training, a **20,000-step fine-tune**, and **140 physical policy evaluations** ran.

**The learned policy remains unreliable on new scenes:** it succeeded in 14/20
training-scene replays and 1/20 held-out color combinations, but 0/20 in each other
generalization condition. This repository teaches the complete pipeline and its
measured limitations. See [the experiment log](docs/experiments.md); scripted
expert success is separate from learned VLA success.

```mermaid
flowchart LR
  D[Physical demonstrations] --> DS[LeRobotDataset] --> FT[SmolVLA fine-tuning]
  FT --> P[SmolVLA]
  I[One RGB image] --> P
  L[Language instruction] --> P
  S[Robot state] --> P
  P --> A[Action chunk] --> IK[Cartesian servo / IK] --> R[Simulated Panda]
  R --> I
  R --> S
```

The policy directly predicts Cartesian gripper poses and opening commands. The
scripted expert reads simulator truth only to create demonstrations. Learned
inference receives pixels, proprioception, and text. The system does not use a
detector, external VLM/LLM planner, MoveIt, reinforcement learning, or a web frontend.

## Installation and environment

Test machine: Ubuntu 24.04, RTX 5070 Ti (16 GB), driver 595.91.07, Miniforge.
The project reuses the existing standalone Isaac Sim installation. Allow disk/network
space for model checkpoints and generated datasets.
The host's existing ROS Jazzy is retained. **ROS is not installed into Conda** and
is not required for this standalone V1.

```bash
./scripts/setup.sh
conda activate vlarobot
python scripts/download_model.py
python scripts/check_environment.py
```

`vlarobot` contains LeRobot, PyTorch, and SmolVLA libraries. Isaac Sim runs using
its existing bundled Python at `~/isaacsim/python.sh`. For another installation:

```bash
export ISAAC_SIM_PATH=/path/to/your/existing/isaacsim
```

A private local subprocess connection transports RGB/state observations and
robot actions between these interpreters. `setup.sh` installs neither Isaac Sim
nor ROS. System Python and unrelated environments are untouched. The detected
simulator is `6.0.0-rc.59+release.41464.5f2772bc.gl`; its installed APIs are checked
against the current official documentation. NVIDIA assets may download on first
use. There is no cloud runtime or mandatory external logging account.

See [environment details](docs/environment.md),
[the implementation plan](docs/implementation_plan.md), and versioned configuration
files under `configs/`. Run all plain `python` commands from the repository root
with `vlarobot` active; shell scripts select the environment automatically.

## Launch the simulator

```bash
./scripts/run_simulation.sh
./scripts/run_simulation.sh --headless --seed 7
python -m src.control.teleoperation
```

The simulator demo uses the explicitly labeled scripted expert. The native viewer
shows the scene, command, seed, step, and status. `outputs/scene.png` captures the
actual rendered camera view. The optional terminal jogger lets you inspect the
same Cartesian action interface manually.

The scene has four colored cubes and four physical open-top boxes. Reset varies
cube positions with reachability bounds, separation constraints, and box exclusion
zones. One RGB camera sees the workspace. Physics runs at 120 Hz, with a 10-Hz
action/observation interface.

## Validate the teacher and collect demonstrations

```bash
python -m src.control.validate_expert --episodes 100
./scripts/collect_dataset.sh --episodes 50
python -m src.dataset.inspect
python -m src.dataset.visualize --episode 0 --play
python -m src.dataset.visualize --episode 17
python -m src.dataset.visualize --episode 33
python -m src.dataset.split
```

The collector refuses to run without a matching expert validation result above
95% on at least 100 randomized episodes. It keeps successful physical
demonstrations, and records the canonical task separately from its sentence.
Review several videos/contact sheets before training.

To collect more, use `--episodes 100`, `250`, `500`, or `1000` and a new `--root`.
`--resume` appends to a compatible collection. Existing roots are otherwise
protected from accidental overwrite.

## Dataset and action format

| Signal | Representation |
|---|---|
| RGB | `observation.images.camera1`, 256×256×3 uint8 at capture |
| State | 14 values: TCP XYZ/RPY, measured gripper width, seven joint angles |
| Language | Natural-language `task` string |
| Action | Seven values: absolute TCP target XYZ/RPY plus opening fraction |
| Outcome | Transition done/success plus separate episode metadata |

Units are metres/radians in the world/robot-base frame; gripper fraction is 0–1.
LeRobot stores video/Parquet plus schema, task indices, episode boundaries, and
normalization statistics. The default split makes 45 training and five validation
episodes from a 50-episode collection. Test scenes use a disjoint seed range;
held-out phrases and source/destination combinations are defined explicitly.

Read [architecture and coordinate frames](docs/architecture.md) and
[dataset/split details](docs/dataset.md) before changing this contract.

## Smoke test and train

Close the simulator to free GPU memory before training.

```bash
python scripts/smoke_test_vla.py
./scripts/train.sh
```

The smoke test loads genuine pretrained weights and runs CUDA inference on a
recorded observation. It reports input shapes, the 50×7 action chunk, VRAM, and
latency. The default training run is a **300-step sanity experiment**, using the
official LeRobot trainer. It is not a claim of a fully competent policy.

Settings are in `configs/training.yaml`. CUDA OOM retries halve batch size down
to one, with separate logs for each attempt. Training logs, memory samples,
checkpoint files, and held-out validation loss stay local. W&B is disabled.

After the sanity run and physical evaluation:

```bash
./scripts/train.sh --config training_longer
./scripts/train.sh --config training_full
```

`configs/training_image512.yaml` repeats the 3,000-step experiment with SmolVLA's
standard 512-pixel internal image size. Camera recordings remain 256 pixels;
upsampling changes the encoder's patch sampling without adding scene detail.

`training_full` uses the verified FP32 parameter-storage correction with BF16
compute and 20,000 steps at batch 32. The completed run took 43.1 minutes on this
machine. These are separate experiments with fresh optimizers; choose a new
`--output` directory to run them again without overwriting their evidence.

See [training details](docs/training.md) for freezing, flow matching, normalization,
OOM behavior, and the difference between starting another fine-tune and resuming
optimizer state.

## Watch the learned policy

```bash
python -m src.inference.run_policy \
  --instruction "Put the green cube in the red box." \
  --checkpoint outputs/train/full/checkpoints/last/pretrained_model
```

Use `--headless` for recording without a window, `--seed` for reproducible scenes,
and `--n-action-steps 1` to replan every control step. The default predicts 50
actions, executes the first five, and then uses fresh observations. It does not
blindly execute one fixed episode trajectory. Videos and action/telemetry traces
are saved under `outputs/videos/`.

To run the configuration of the full model's successful held-out combination:

```bash
python -m src.inference.run_policy --headless --seed 300005 \
  --instruction "Take the yellow cube to the red box."
```

The actual recording is `outputs/videos/evaluation/composition_0005.mp4`;
its [contact sheet](docs/media/vla_full_composition_success.jpg) shows the grasp,
transport, release, and retreat. To inspect the earlier 3,000-step success:

```bash
python -m src.inference.run_policy --headless --seed 100000 \
  --instruction "Pick up the red object and put it in the green container." \
  --checkpoint outputs/train/fp32/checkpoints/last/pretrained_model
```

The recorded [successful VLA contact sheet](docs/media/vla_first_success.jpg)
and [failed VLA contact sheet](docs/media/vla_3000step_failure.jpg) show actual
simulator frames. One success does not establish a reliable policy.
Seeds fix the scene layout and model sampling, but RTX rendering is not guaranteed
pixel-identical across runs or GUI/headless modes. A GUI trial with the same seed
failed; the result is retained in the experiment log. The full model's GUI trial
on the yellow → red example also failed its grasp. Native-viewer synchronization
passed; the recorded headless successes do not guarantee repeatable task success.

## Evaluate and analyze failures

```bash
./scripts/evaluate.sh
./scripts/evaluate.sh --episodes 50 --conditions position language composition \
  --output outputs/evaluation_50
python -m src.evaluation.analyze_failures --root outputs/evaluation
python scripts/plot_results.py --training outputs/train/sanity outputs/train/longer \
  --evaluation outputs/evaluation
```

Default evaluation conditions are training-scene replay, unseen positions,
unseen phrases, held-out color combinations, changed lighting, camera perturbation,
and cube scale. The output includes per-episode `results.csv`, Wilson confidence
intervals, a generated results table, videos, and failure traces.

Success requires the target cube to rest fully inside the right box, with the
gripper open and withdrawn for a stable interval. Failure classification is
conservative: wrong object, failed approach/grasp, wrong destination, dropped or
misplaced object, or timeout. Videos help distinguish ambiguous causes.

See [inference/evaluation](docs/inference.md) and
[actual experiment results](docs/experiments.md). Bad outcomes remain in reports.
No learned success rate has been asserted before completing the corresponding run.

Generated datasets, weights, and videos stay in ignored local folders. Compact
measurements and actual contact sheets are versioned under `docs/results/` and
`docs/media/` so the experiment record remains part of the repository.

Measured final results: 20 trials per condition, 20,000-step checkpoint, five
executed actions per chunk, and a 240-step episode limit.

| Experiment | Successes | Success rate | Wilson 95% CI |
|---|---:|---:|---:|
| Training-scene replay | 14/20 | 70% | 48.1%–85.5% |
| Unseen positions, recorded wording | 0/20 | 0% | 0%–16.1% |
| Held-out paraphrases | 0/20 | 0% | 0%–16.1% |
| Held-out color combinations | 1/20 | 5% | 0.9%–23.6% |
| Lighting variation | 0/20 | 0% | 0%–16.1% |
| Camera perturbation | 0/20 | 0% | 0%–16.1% |
| Cube scale variation | 0/20 | 0% | 0%–16.1% |

![Measured closed-loop results](docs/media/evaluation.png)

Test conditions use paired fresh scene seeds 300000–300019; training replay is
a separate memorization diagnostic. A 0/20 result does not prove zero underlying
success probability. Because the position baseline is already zero, the other
zero rows cannot establish how much a wording or visual change hurts performance.
The earlier development pilots and failed attempts remain in the experiment log.

Across 140 trials, the outcome heuristics recorded 96 approach failures, 15 failed
grasps, and 14 wrong-object lifts. All 140 traces and 15 representative videos
were retained. The primary limitation is initial object approach on new scenes;
the data contains only four to six demonstrations per canonical pair.

## Troubleshooting

| Symptom | Check |
|---|---|
| ROS Python/plugin imports inside Conda | Reactivate the environment after setup; its `PYTHONPATH` is isolated. No ROS reinstall is needed. |
| CUDA unavailable / unsupported architecture | Run `scripts/check_environment.py`; use the pinned CUDA 12.8 PyTorch build for this Blackwell GPU. |
| First Isaac startup is slow | Inspect `outputs/logs/simulation.log`; extension loading, asset downloads, and shader compilation can take time. |
| Isaac exits before connecting | Read its log; verify the existing Isaac launcher, `vlarobot`, and the `.kit` dependencies. Set `VLAROBOT_SIM_PYTHON` for a custom interpreter. |
| Asset service unavailable | Set `VLAROBOT_FRANKA_USD` to a complete local Panda asset tree. |
| Empty camera image | Check synchronous rendering and camera placement; do not train on blank observations. |
| Expert gate refuses collection | Re-run validation after physics/control edits; inspect failed expert traces/videos. |
| Model moves in the wrong scale/frame | Verify action units, saved normalizer, state order, and pretrained feature overrides. |
| GPU OOM during training | Close Isaac; the runner retries smaller batches. Inspect preserved attempt logs. |
| Low loss, bad manipulation | Inspect dataset alignment and failures; expand coverage/recovery data and evaluate closed loop. |
| Existing experiment/dataset directory | Choose a new output path or the documented resume workflow; results are preserved by default. |

## Tests

```bash
python -m pytest -q
python -m ruff check src scripts tests
```

These tests cover contracts and the real LeRobot writer API. The tiny synthetic
writer fixture is temporary test data, never a substitute for physical robot
demonstrations. Physics/expert/model/training checks are separate runtime milestones.
The final software/API suite has 15 passing tests.

## Limitations and next steps

This is a small, controlled pick-and-place experiment: one camera, four geometric
objects, fixed bin positions, narrow gripper orientation, and scripted successful
demonstrations. It has no guarantee of recovery, collision avoidance, arbitrary
language understanding, real-time hardware performance, or sim-to-real transfer.

Study [the learning notes](docs/learning_notes.md) for a walkthrough from pixels
and language to motor commands. They cover behavioral cloning, action chunks,
flow matching, embodiment, failure modes, and why this differs from an LLM/planner
system. Future versions can add manipulation tasks, Isaac Lab randomization,
ROS 2 using the existing Jazzy setup, real hardware, and eventually mobile and
industrial systems. Those extensions are documented, not implemented in V1.

## Primary references

- [LeRobot documentation](https://huggingface.co/docs/lerobot)
- [SmolVLA training guide](https://huggingface.co/docs/lerobot/smolvla)
- [Official base checkpoint](https://huggingface.co/lerobot/smolvla_base)
- [SmolVLA technical report](https://huggingface.co/papers/2506.01844)
- [Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
