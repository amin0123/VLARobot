# Closed-loop inference and evaluation

New evaluation runs save `run_config.json` with a SHA-256 hash of the exact model
weights, the environment configuration, action prefix, episode limits, and seed
range. Per-episode records include settled initial object poses. Camera, lighting,
and scale perturbations can be regenerated from the recorded condition/seed and
configuration. Each condition must be listed once, so duplicate CLI entries
cannot inflate the episode count by repeating the same trials.
Training-scene replay reads the dataset location saved by that checkpoint's
trainer. If the repository/dataset has moved, pass `--training-root` explicitly.

```bash
python -m src.inference.run_policy \
  --instruction "Put the green cube in the red box." \
  --checkpoint outputs/train/fp32/checkpoints/last/pretrained_model
```

This launches the native Isaac viewport and an episode-status panel. `--headless`
disables the window while retaining RGB rendering. A recording and trace are
written under `outputs/videos/`. Recordings include instruction, step, and outcome
outside the policy image. Overlay text is never sent to SmolVLA.

The input allow-list supplies one RGB image, 14 measured state values, and the
original instruction. The official processor and action queue are loaded from
the checkpoint. `--n-action-steps 1` recomputes a chunk each control step; the
default is five. Reset clears the model queue and processors between episodes.

An optional source/destination label tells the evaluator how to score unusual
wording. The policy receives neither these labels nor coordinates. For the
controlled two-color instructions, a small parser extracts the labels for scoring.
The parser does not generate motions or replace the text presented to the model.

## Automatic evaluation

```bash
./scripts/evaluate.sh
./scripts/evaluate.sh --episodes 50 --conditions position language composition \
  --output outputs/evaluation_50
```

Each condition gets distinct episodes within it. Position, language, composition,
lighting, camera, and scale use the same evaluation seed sequence so perturbation
comparisons have paired layouts. SmolVLA's flow-noise generator is also seeded
per episode; `policy_seed` is recorded alongside the scene seed. `train_replay` intentionally uses training
episodes and is labeled separately. It measures memorization, not generalization.

The success criterion checks all eight cube corners against the target container's
interior, a resting height near the floor, low object velocity, open fingers,
and a withdrawn TCP. It must hold for eight 10-Hz steps. Looking successful in
one frame, hovering above a box, or holding the cube over the rim does not pass.

Results are written after every episode:

- `results.csv`: condition/task/instruction/seed, initial scene, success, duration,
  episode length, video path, and failure category.
- `summary.json`: per-condition counts, rates, and Wilson 95% intervals.
- `report.md`: a table generated solely from completed rollouts.
- `traces/`: evaluator telemetry, predicted actions, and inference latency.
- `outputs/videos/<evaluation-name>/`: actual recorded RGB episodes.

The default retains the first two episodes of each condition plus the first
success and first failure, if those occur later. Extra candidate recordings are
removed after their result is known. Increase `--videos` to retain more, or
replay an interesting seed interactively. Reports count every
completed policy attempt, including failures. Interrupted runs retain completed
rows and should not be silently merged with a different checkpoint.

## Failure interpretation

Automatic labels are conservative event heuristics: approaching no target,
approaching but never lifting, lifting a wrong object, placing near a wrong
container, dropping/misplacing after lift, or a placement timeout. They are
diagnostics, not proofs of an internal semantic failure. Collision, oscillation,
action drift, and language misunderstanding may require manual video/action
inspection; do not fabricate a cause when available measurements cannot establish
it. A cube slipping during transport can follow an otherwise correct language
selection.

The evaluation table's confidence intervals describe independent randomized
episodes within a condition. Conditions are paired, so their results are correlated.
Comparing checkpoints formally calls for paired statistics and more seeds. A
small sample with zero successes still has a nonzero upper confidence bound.

The native viewer and video are for inspection. Simulator time is advanced at
10 action intervals per simulated second; wall-clock latency can be slower or
faster and is recorded separately.

## Inspecting measured artifacts

```bash
python -m src.evaluation.analyze_failures --root outputs/evaluation
python scripts/plot_results.py --training outputs/train/sanity outputs/train/longer \
  --evaluation outputs/evaluation
```

The analysis tool counts recorded failure labels, reports measured target lift
and approach distance, and extracts temporal contact sheets from real videos.
The plotting tool uses saved loss samples and evaluation counts only.
