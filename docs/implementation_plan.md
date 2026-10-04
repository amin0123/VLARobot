# Implementation plan

Started 2026-10-04 and updated through the completed runtime milestones below.
An implemented feature is not considered validated until run.

## Architecture and scope

The user's final instruction selects **Isaac Sim**, superseding the earlier
MuJoCo preference and Isaac Sim exclusion. V1 uses standalone Isaac Sim Python,
a Franka Panda arm with parallel gripper, one RGB camera, four colored cubes,
and four physical, open-top containers. No ROS, Isaac Lab, detector, LLM planner,
reinforcement learning, or cloud service is part of the control path.

```mermaid
flowchart LR
  I[RGB camera] --> P[SmolVLA pretrained policy]
  L[Natural language] --> P
  S[Robot proprioception] --> P
  P --> A[Action chunk]
  A --> C[Robot servo / kinematic controller]
  C --> R[Isaac Sim Franka]
  R --> I
  R --> S
```

## Implementation choices

- Learning libraries: dedicated Conda environment named `vlarobot`; leave system
  Python and existing environments alone. Isaac uses its existing bundled Python.
- Simulator: the existing standalone Isaac Sim installation, checked against current
  NVIDIA documentation and LeRobot dependencies. Use CPU PhysX for this single
  scene if it improves reproducibility; RTX rendering supplies the RGB image.
- Robot: NVIDIA's Franka Panda USD asset. Articulated joints and contact physics,
  with real grasping rather than attaching/teleporting objects during control.
- Action selected before collection: seven values for absolute TCP XYZ/RPY and
  gripper opening, converted to joint targets by deterministic IK. The initial
  proposal considered incremental poses; absolute targets were selected for
  direct inspection and to avoid accumulating noisy deltas. See `architecture.md`
  for units, clipping, and the exact implementation contract.
- Observation: one RGB image, 14 measured state values (TCP XYZ/RPY, total gripper
  width, and seven arm angles), and the instruction. Object poses and canonical
  task IDs are excluded from the policy input.
- Timing: 10 Hz action/observation interface and 120 Hz physics, with explicit
  timestamps and documented units. The camera captures 256×256 RGB.
- LeRobot: use the current official `LeRobotDataset` API and official SmolVLA
  preprocessing/training utilities. Pin the tested release or commit. Dataset
  frames contain the observation **before** the recorded action is executed.
- Expert: finite-state pick, grasp, lift, transfer, place, release, retreat.
  Simulator truth is available only to this teacher and the evaluation metric.
- Data: at least 100 randomized expert validation episodes, targeting >95%
  success, then 50 successful demonstrations for the first controlled experiment.
  Canonical source/destination, instruction, seed, and outcome stay in metadata.
- Splits: disjoint episode seeds; separate held-out language templates and
  source/destination combinations; later lighting, camera, and scale changes.
- Training: first load the official `lerobot/smolvla_base` checkpoint and measure
  a CUDA inference pass. Then a short fine-tune, verify loss/checkpoints, then a
  larger run as resources permit. Retry CUDA OOM with a reduced batch size. Keep
  logs local. Do not label a short sanity run as a fully trained policy.
- Inference: execute a configurable prefix of predicted chunks and observe again;
  use exactly the dataset's state/action semantics and saved normalization.
- Evaluation: physical containment, stable placement and released gripper;
  per-episode CSV/JSON, Wilson confidence intervals, recordings, and conservative
  failure labels. Never substitute expert actions in a VLA rollout.

## Milestones and gates

1. Inspect hardware; choose, install, and record compatible dependencies.
2. Launch/render the robot scene; verify reset, camera, and actions.
3. Make the expert reliable on 100 randomized episodes before collecting data.
4. Export, inspect, and visually review a real LeRobot dataset.
5. Verify pretrained SmolVLA shapes, CUDA, preprocessing, and forward inference.
6. Run short training, then longer training; record actual resource use/results.
7. Run closed-loop evaluation, generalization slices, videos, and failure analysis.
8. Finish educational docs and commands; run checks and commit each major stage.

## Expected technical risks

- Isaac Sim and LeRobot may require conflicting Python/PyTorch/NumPy versions.
  Resolve compatible pins first; if unavoidable, isolate the simulator process
  from policy dependencies using local IPC, without changing the VLA mapping.
- RTX 5070 Ti is Blackwell: PyTorch must include its CUDA architecture. The
  `nvidia-smi` CUDA version is driver capability, not a installed CUDA toolkit.
- RTX rendering and policy training contend for 16 GB VRAM. Collect first, close
  the simulator, then train; use conservative image sizes and batch sizes.
- USD asset downloads, extension startup, shader compilation, and first model
  download can be slow. Cache assets and retain logs; distinguish network errors
  from model or physics errors.
- Grasp friction, cube spacing, reachable poses, and release stability can make
  the expert unreliable. Fix physical/control causes before collecting.
- SmolVLA's pretraining embodiment differs from Panda. Fifty demonstrations are
  an experiment, not evidence of reliable language or compositional generality.
- Estimated long-run cost must come from the short benchmark. Any incomplete
  expensive run is reported honestly, with reproducible continuation commands.

## Official references to check before implementation

- https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html
- https://docs.isaacsim.omniverse.nvidia.com/latest/index.html
- https://huggingface.co/docs/lerobot/installation
- https://huggingface.co/docs/lerobot/smolvla
- https://huggingface.co/docs/lerobot/lerobot-dataset-v3
- https://github.com/huggingface/lerobot
- https://huggingface.co/lerobot/smolvla_base

## Status

The full pipeline, 20,000-step training, 140-episode evaluation, and educational
documentation are complete; new-scene policy reliability is poor and reported
explicitly. The initial implementation
mistakenly began a second pip installation of Isaac Sim; this was stopped after
the user clarified that Isaac Sim is already installed. The project now reuses
`/home/amin24/isaacsim/python.sh` (6.0.0-rc.59) and does not install a simulator
or ROS. NumPy/packaging conflicts between current pip Isaac Sim and LeRobot
still motivate separate interpreters; the existing simulator supplies its own.
See `environment.md` for the decision and `experiments.md` for executed milestones.

## Milestone evidence

| Milestone | Current evidence |
|---|---|
| Isolated learning environment | CUDA computation, TorchCodec import, `pip check`; existing Isaac/ROS reused |
| Robot and randomized scene | Real RGB render and synchronized-state integration check |
| Reliable scripted expert | 100/100 randomized physical successes |
| LeRobot demonstrations | 50 episodes, 3,914 frames, v3.0 MP4/Parquet |
| Dataset visualization | Episodes 0, 17, 33 decoded and visually reviewed |
| SmolVLA smoke test | CUDA output [1, 50, 7], 78.9 ms warm inference |
| Sanity fine-tune | 300 official trainer steps; checkpoints and validation loss recorded |
| Full fine-tune | 20,000 steps/batch 32 completed in 43.1 minutes; checkpoints and held-out loss saved |
| Closed-loop VLA | Direct RGB/text/state control produced physical successes; failures retained |
| Generalization suite | 140 completed trials: 14/20 training replay, 1/20 held-out combinations, 0/20 in each other condition |
| Educational documentation | Complete pipeline guides, measured tables, uncertainty, failure analysis, and actual success/failure frames |

Exact measurements, failed attempts, and artifact paths are in `experiments.md`.
