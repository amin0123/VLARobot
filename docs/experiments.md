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

## Outstanding runtime gates

Isaac Sim launch/render, randomized expert trials, real dataset collection and
visual review, pretrained forward pass, training, and VLA rollout evaluation are
not yet recorded here. No task success rate is asserted for them.
