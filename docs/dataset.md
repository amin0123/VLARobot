# Demonstrations and dataset

First validate the teacher, then collect. The collector requires more than 95%
success on at least 100 randomized episodes. A fingerprint of the physics,
controller, expert, sampler, and metric invalidates that gate if their code changes.

```bash
python -m src.control.validate_expert --episodes 100
./scripts/collect_dataset.sh --episodes 50
python -m src.dataset.inspect
python -m src.dataset.visualize --episode 0 --play
python -m src.dataset.visualize --episode 17
python -m src.dataset.visualize --episode 42
python -m src.dataset.split
```

## What a demonstration contains

Each frame has one uint8 RGB observation, 14 float state values, seven float
Cartesian target/gripper actions, the natural-language `task`, and transition
`next.done`/`next.success` fields. The official writer supplies timestamps, frame
indices, episode indices, and task indexing. Images are sampled before the
associated action; terminal success belongs to the transition after that action.

Only successful episodes enter behavioral-cloning data. Failed collection
attempts are reported and discarded through the writer API. Expert validation
retains all attempted episode outcomes so its reported success rate is unbiased
by successful-only collection. This filtering is not a license to hide failures
in policy evaluation, where every attempted rollout counts.

The local format is the official LeRobot v3 family: Parquet tabular signals,
compressed MP4 RGB streams, metadata describing boundaries, schema, and statistics.
Use `meta/info.json` and the installed `LeRobotDataset` implementation as the
format authority; do not assume one MP4/Parquet file per episode. Writer
`finalize()` is mandatory to close Parquet footers and pending video work.

Project metadata is kept separately:

- `meta/collection.jsonl`: canonical source/destination, exact sentence/template,
  seed, accepted episode index, success, length, initial scene.
- `meta/project.json`: action/state contract, environment configuration, expert gate.
- `meta/inspection.json`: dimensions, frame/episode counts, FPS, language coverage.
- `meta/collection_summary.json`: collection duration and completion status.

None of the sidecar object coordinates are training features. The policy input
factory admits only `observation.state`, `observation.images.camera1`, and `task`.

## Controlled variation

Four distinct cube colors and four container colors are visible together. Cube
positions vary by seed with a minimum center spacing of 9.5 cm and exclusion
zones around bins. Containers stay in fixed positions for V1. Cubes start
axis-aligned; there is no random object orientation, texture clutter, or multi-task
manipulation in the first experiment.

The five training templates vary verbs (`put`, `move`, `place`, `pick up`, `take`),
object nouns (`cube`, `block`, `object`), and destination nouns (`box`, `container`).
Three held-out templates live in `src/tasks.py`. Templates and canonical task
labels are separate to distinguish phrase variation from task variation.

Nine different-color source/destination pairs are available for training. The
held-out combinations are green → blue, blue → yellow, and yellow → red. Same-color
source/destination tasks are outside the initial training/evaluation distribution.

## Splits and leakage

`src.dataset.split` uses LeRobot's official `split_dataset` utility to materialize
train/validation datasets with separate aggregated statistics. For 50 collected
episodes, the default is 45 train and five validation episodes. A seeded selection
preserves at least one training episode for each canonical task represented in
the source dataset. These are whole episode splits, never random individual frames.

The official trainer's exact-sentence stratification can hold out a large fraction
of a tiny paraphrased dataset, so V1 uses explicit split indices instead. Training
normalization comes only from the train dataset. Validation uses the checkpoint's
saved training normalizer, not validation-derived statistics.

Final test seeds begin at 300000, independently of development rollout seeds
100000–100002 and collection seeds beginning at
1000 and expert validation seeds beginning at 10000. Visual perturbation conditions
share evaluation seeds so comparisons use paired scenes. The `train_replay`
condition intentionally reuses training scenes and is reported as such; it is a
memorization diagnostic, not generalization evidence.

The final evaluation uses `instruction_sampling: seen`: position, lighting,
camera, and scale all use an exact sentence recorded in training for each canonical
pair. Language evaluation changes to held-out templates on the same paired scenes;
composition uses the three held-out color pairs. This avoids mixing a new wording
combination into the position-only result.

The early development pilots sampled familiar templates and could create exact
sentences absent from the small training split. They are preserved as development
results. `--instruction-sampling templates` reproduces that earlier sampling
rule. The final protocol version and selected rule are recorded in `run_config.json`.

## Scale up

```bash
./scripts/collect_dataset.sh --episodes 100 --root datasets/pick_place_100
./scripts/collect_dataset.sh --episodes 250 --root datasets/pick_place_250
./scripts/collect_dataset.sh --episodes 500 --root datasets/pick_place_500
./scripts/collect_dataset.sh --episodes 1000 --root datasets/pick_place_1000
```

Choose a new root to preserve earlier experiments. `--resume` appends to an
existing collection, checks metadata alignment, and skips accepted seeds. The
requested episode count is the desired total, not the number to append. Before
opening the writer it checks scene configuration, controller fingerprint,
state/action schema, FPS, robot type, split, episode indices, and unique seeds.
It preserves the original project metadata and returns without launching Isaac
when the compatible dataset already has enough episodes.

Appending after a controller/physics revision requires a new root. This is
deliberately stricter than deciding whether an old dataset is still useful for
training: the original 50-episode dataset remains valid after the verified scale
reset fix, but its historical controller fingerprint is retained rather than
rewritten to permit mixed provenance.

Review
language/task balance and episode videos before increasing training time. Fifty
episodes spread over nine tasks and five phrases are a deliberately small
baseline, not a large multi-task dataset.

In the actual 45-episode training split, all 31 collected sentences remain
represented: **20 sentences have one demonstration, eight have two, and three
have three**. Each canonical pair has four to six demonstrations. This sparse
coverage can permit associations between a sentence and particular scene layouts;
it does not force reliable localization or source/destination binding. That is a
possible shortcut to investigate, not a proved internal explanation of the
model's failures. Increasing repetitions across positions for each canonical
pair and balancing wording are useful next controlled experiments.

Reference: [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
and the pinned `lerobot.datasets.dataset_tools.split_dataset` source.
