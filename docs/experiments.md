# Experiment log

This document records actual results only. A source file existing is not proof
that its runtime path works. Runtime results will be added at each completed gate.

## 2026-10-04 — machine and dependency inspection

Ubuntu 24.04.5, RTX 5070 Ti (16,303 MiB), driver 595.91.07, 31 GiB RAM, about
650 GiB free disk. Selected Python 3.12, PyTorch 2.11.0 + CUDA 12.8, LeRobot 0.6.1,
and the existing standalone Isaac Sim 6.0.0-rc.59. An initial redundant pip
installation was stopped and its task-created simulator environment removed.
The final design uses one Conda environment for learning libraries and Isaac
Sim's existing bundled Python. Existing ROS Jazzy remains on the host.

Measured CUDA check: an 8×8 matrix of ones multiplied by itself produced 8.0
on the GPU. `torch.cuda.is_available()` returned true and identified the RTX
5070 Ti. Detailed package/architecture output: `outputs/environment.json`.

## Contract tests

Eight tests passed after isolating inherited ROS Python paths. Tests covered
1,000 randomized scene layouts, whole-cube containment/resting height, Wilson
interval edge cases, task/phrase split separation, metric parsing, and the policy
input allow-list. These are software/geometry checks, not physical grasp trials.

A separate small synthetic LeRobot round-trip fixture caught a codec API change:
LeRobot 0.6.1 accepts `h264` rather than `libx264` in `RGBEncoderConfig`. The collector
uses the supported name. Synthetic API fixtures are temporary test files and are
never used as demonstrations, videos of robot behavior, training data, or results.

## Pretrained checkpoint load

The official SmolVLA checkpoint loaded on CPU with 450,046,176 parameters;
99,880,992 were trainable under the default frozen-backbone configuration.
`load_vlm_weights=False` avoids a redundant backbone download while the complete
SmolVLA checkpoint supplies its pretrained weights. This was a weight-loading
check, not an inference or training result. Log: `outputs/logs/pretrained-load.log`.

## Existing Isaac Sim scene and synchronization

The existing 6.0.0-rc.59 runtime renders the Panda, four cubes, and four open
containers with a 256×256 RGB camera. Its first launch took approximately 146 s;
subsequent warm launches took about 4.5 s before scene creation. Robot state has
14 components and actions have 7. Actual images: `outputs/scene.png` and
`outputs/scene_after_actions.png`.

`python scripts/check_simulation.py` passed against the real simulator. Capturing
an extra image changed the robot state by **0.0**; replaying 12 identical actions
with and without rendering also produced a maximum state difference of **0.0**.
The image changed after movement (mean absolute pixel difference 5.893). This
checks that RGB capture is synchronized and does not silently advance physics.

Two camera integration problems were fixed: the installed USD camera has a
double-precision orientation attribute, and render-only ticks did not trigger
its RGB annotator. The code now preserves the attribute type and uses official
Replicator synchronous capture with `delta_time=0.0`.

## Scripted expert development

The first two recorded diagnostic episodes failed: the teacher waited forever
for a placement pose within 6 mm, while the PD-controlled arm had a roughly 1 cm
steady tracking error. The cube was grasped and transported, but not released.
The teacher now releases just above the container rim (to give open fingers
clearance) and uses a physically attainable approach tolerance.

A subsequent 10-episode pilot achieved **9/10**. The remaining episode stopped
in the descent phase: its 7.4 mm pose error exceeded the 6 mm transition tolerance.
The descent tolerance was changed to 10 mm, within the grasp-pad region. A fresh
100-episode validation is required before demonstration collection. Diagnostic
logs and videos remain under `outputs/expert_debug/` and `outputs/expert_pilot/`.
These figures describe the scripted teacher, not SmolVLA.

## Scripted expert gate — passed

Fresh validation on seeds **10000–10099** achieved **100/100 successful episodes**,
with a Wilson 95% confidence interval of **96.3%–100%**. Tasks cycle through all
nine training color pairs and sample controlled language variations. These seeds
are separate from demonstration collection and learned-policy evaluation seeds.

Success requires the entire target cube inside the correct physical container,
resting near its floor, moving slower than 2 cm/s, and released with the gripper
retreated for eight consecutive control steps. No object attachment or pose
teleport is used during manipulation. Machine-readable results and two real
expert videos are under `outputs/expert/`. The collector checks the result and
an environment/controller fingerprint before writing demonstrations.

## Native viewer and dependency correction

The native GUI demo completed successfully in 76 control steps (7.6 simulated
seconds), showing the current instruction and episode status. Log:
`outputs/logs/gui-demo.log`; actual camera frame: `outputs/gui_scene.png`.

Dataset collection exposed a TorchCodec packaging mismatch: the generic PyPI
0.11.0 wheel requested `libnvrtc.so.13`, while our PyTorch wheel uses CUDA 12.8.
The collector continued through the explicitly selected supported PyAV backend.
The CUDA 12.8 TorchCodec wheel then exposed an optional missing NVIDIA NPP
library. V1 uses CPU video decoding, so the final compatible choice is the
official `torchcodec==0.11.0+cpu` wheel. Its import, the CUDA PyTorch computation,
and `pip check` all pass. This avoids adding an unused GPU video-decoder stack;
SmolVLA computation still uses CUDA. The environment checker now imports the
codec, since metadata-only version checks missed the shared-library mismatch.

## LeRobot demonstration dataset — passed

Collected **50/50** successful episodes on seeds **1000–1049**, without a rejected
trial, in **478.8 s**. The official v3.0 dataset contains **3,914 frames at 10 Hz**,
31 distinct instructions, nine canonical color pairs, and 50 unique scene seeds.
Episode duration is 7.0–9.1 s (mean 7.828 s). The state width is 14 and action
width is seven. Inspection: `datasets/pick_place_50/meta/inspection.json`.

Visual review of decoded dataset episodes **0, 17, and 33** confirmed approach,
physical closure, lift, transport, release, and retreat. Their real MP4 replays
and contact sheets are under `outputs/dataset_preview/`. Episode 33 also shows
a small incidental displacement of a neighboring cube; the task metric does not
penalize that. Expert success is not evidence of collision-free manipulation or
preserving every distractor's position.

The official split utility created **45 training episodes / 3,526 frames** and
**5 validation episodes / 388 frames**, with disjoint seed sets. Training retains
all nine canonical pairs. Validation normalization will come from the training
checkpoint. `datasets/splits/manifest.json` records exact indices and seeds.

## Pretrained CUDA forward pass — passed

`python scripts/smoke_test_vla.py` loaded the official pretrained SmolVLA weights
and consumed a real training-split observation. The output was finite with shape
**[1, 50, 7]**. Warm chunk inference took **78.9 ms** on the RTX 5070 Ti; PyTorch
peak allocated memory was **1,188.9 MiB**. The model has 450,046,176 parameters,
with 99,880,992 trainable under the selected frozen-backbone configuration.
Full input shapes and measurements: `outputs/smoke_test_vla.json`.

These are single-example smoke-test measurements after warm-up, not an end-to-end
control benchmark or a demonstration of task competence. Base outputs were not
executed as evidence of success.

## Outstanding runtime gates

The 300-step sanity fine-tune, longer training run, and VLA rollout evaluation
are not yet recorded here. No task success rate is asserted for them.
