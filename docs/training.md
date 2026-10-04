# Fine-tuning SmolVLA

Run dataset inspection and visual review before training. A decreasing loss
confirms optimization, not physical task success.

```bash
conda activate vlarobot
python scripts/download_model.py
python scripts/smoke_test_vla.py
./scripts/train.sh
```

The smoke test uses a real recorded RGB/state/language example, changes the
checkpoint's feature contract to this embodiment, loads pretrained weights,
performs a warm-up and a timed CUDA inference, denormalizes an action, and verifies
finite output of shape `[1, 50, 7]`. Its JSON records shapes, parameters, VRAM, and
latency. It does not claim that the unadapted pretrained actions solve this task.

## What is trained

The official base is `lerobot/smolvla_base`. The default SmolVLA fine-tuning setup
freezes the vision-language backbone and trains the action expert and state/action
projections according to the installed upstream configuration. The action expert
learns this Panda's Cartesian control distribution from expert demonstrations.
The original state/action shape is adapted through SmolVLA's 32-dimensional
internal padding, not by replacing the model with another architecture.

`load_vlm_weights=False` during checkpoint construction avoids downloading the
backbone's separate weights redundantly. The **full pretrained SmolVLA checkpoint
is still loaded**, including backbone and action expert. This flag does not mean
we train from random initialization.

The action loss is conditional flow matching: the learner is given noisy target
action chunks at sampled flow times and learns a vector field toward expert
actions. During inference, iterative integration transforms noise into a chunk
conditioned on the current image, instruction, and state. See `learning_notes.md`.

## Runner and resource controls

`src/training/train_smolvla.py` invokes the installed official
`lerobot.scripts.lerobot_train`. It does not replace the training loop. Configuration
in `configs/training.yaml` controls batch size, steps, learning rate, warm-up,
checkpoint/log intervals, image size, chunk size, execution prefix, device,
mixed precision, workers, and seed.

The default is a 300-step sanity run. Each invocation uses a new output directory
and refuses to overwrite an experiment. The runner records the exact command,
requested configuration, official trainer log, parsed loss samples, total elapsed
time, and `nvidia-smi` memory/utilization samples. W&B is disabled; no account is
required. GPU memory includes other processes such as the desktop, whereas the
smoke test also reports PyTorch's allocation accounting.

On this pinned LeRobot release, the small `official_entrypoint.py` launcher sets
PyTorch's CUDA autocast dtype to BF16 before calling the official trainer. The
SmolVLA config has `use_amp` but no `dtype` field, so upstream otherwise chooses
CUDA's FP16 default. That created a GradScaler unable to unscale the checkpoint's
BF16 gradients in our first attempted step. BF16 autocast avoids that mismatch
and uses the same precision as our inference smoke test. No upstream library
files or training-loop implementation are patched. `mixed_precision: false`
disables the upstream autocast selection.

The precision diagnostic in `configs/training_fp32.yaml` additionally sets
`trainable_dtype: float32`. Its entrypoint wraps the official policy factory to
convert trainable parameter storage before optimizer construction; frozen weights
retain their dtype, and compute still uses BF16 autocast. AdamW then stores its
moments in FP32. This is distinct from changing every operation to FP32. The
earlier configurations omit the setting to preserve their original experiments.
See the measured comparison in `experiments.md`; dtype choice alone is not a
guarantee of manipulation competence.

SmolVLA's nested Transformers constructor can also narrow saved FP32 expert
weights back to BF16 at reload. `src/training/precision.py` restores those FP32
tensors after the official load, in training and inference. This preserves their
saved values and the precision used by residual operations. It does not downcast
frozen weights that upstream already constructs in FP32. This compatibility step
is covered by an exact-value checkpoint round-trip test.

CUDA OOM triggers a bounded retry at half the batch size, ending at batch size 1.
Retries use fresh child processes to release allocations and retain failed logs.
They restart from the selected pretrained checkpoint; they do not claim optimizer
continuity or an unchanged effective batch size. Non-OOM failures stop with the
log location. Close Isaac Sim before training to free its RTX memory.

After training, the runner computes held-out flow-matching loss using the saved
normalizer. Closed-loop simulator success is evaluated separately. `checkpoints`
under the run directory is a link to the successful attempt; it contains the
official model, processor files, optimizer/scheduler state, and train configuration.
That convenient link is created only after the run completes. During training,
intermediate checkpoints are under `attempt_00_batch_32/checkpoints/` (with the
actual attempt number and batch size), such as `010000/pretrained_model`.

## Longer experiments

After the sanity run passes and its videos have been assessed:

```bash
./scripts/train.sh --config training_longer
./scripts/evaluate.sh \
  --checkpoint outputs/train/longer/checkpoints/last/pretrained_model \
  --output outputs/evaluation_longer
```

This is a larger local experiment, not equivalent to the upstream 20k-step
reference recipe. Record actual steps, batch size, data, duration, and results.
Use the saved `train_config.json` and `--resume=true` through the project's
compatibility entrypoint to resume the official optimizer/scheduler and data
order. For example, resume the interrupted portion of a full run in a new output
directory:

```bash
python -m src.training.official_entrypoint \
  --config_path=outputs/train/full/checkpoints/015000/pretrained_model/train_config.json \
  --resume=true --output_dir=outputs/train/resumed_from_15000
```

This retains the BF16 autocast, FP32 checkpoint restoration, and pinned tokenizer
setup. Selecting a model through `train.sh --checkpoint` instead begins a new
fine-tune with a fresh optimizer. Increasing the total step limit while resuming
keeps the saved scheduler state; it does not create a fresh learning-rate schedule.
The wrapper's batch-halving retry applies to fresh fine-tunes, not this direct
optimizer-resume command.

The completed full experiment uses FP32 trainable parameter storage, BF16
computation, 20,000 steps, batch 32, 1,000 warm-up steps, and checkpoints every
5,000 steps:

```bash
./scripts/train.sh --config training_full
./scripts/evaluate.sh \
  --checkpoint outputs/train/full/checkpoints/last/pretrained_model \
  --output outputs/evaluation
```

On this machine it took 43.1 minutes, with no OOM retry. It sampled 640,000
training frames with replacement from 3,526 frames in 45 demonstrations. Repeating
the dataset 181.51 times does not create new scene coverage. Training loss fell
to 0.002 while held-out flow loss rose to 0.09645; inspect the physical evaluation
in `experiments.md` before judging model quality. Use `--output` to preserve the
existing completed experiment when running another training job.

The [current official guide](https://huggingface.co/docs/lerobot/smolvla), checked
2026-10-04, illustrates a 20,000-step, batch-64 run and emphasizes repeated
demonstrations for each introduced variation. Its 50-episode example repeats five
positions ten times each. Our 50 episodes span nine color pairs and continuously
randomized positions, so they give much less repetition per variation. Fifty is
a controlled starting dataset, not a universal sufficiency threshold.

## Separating imitation error from physical success

An optional offline diagnostic compares the first predicted Cartesian action to
the recorded expert target on sampled frames:

```bash
python -m src.evaluation.action_diagnostics \
  --checkpoint outputs/train/longer/checkpoints/last/pretrained_model \
  --root datasets/splits/train --output outputs/action_diagnostics_longer.json
```

It reports XYZ error in metres and gripper-command error. It also replaces the
image with black pixels or the instruction with empty text, holding the flow
noise seed fixed. A changed prediction establishes input sensitivity only. These
counterfactual inputs are outside the training distribution; the experiment does
not establish correct color recognition or language understanding. Simulator
rollouts remain the primary task metric.

Use `--sampling episode_start --samples 45` to measure the initial approach across
all 45 training episodes. An average over an entire demonstration can conceal
large initial localization errors: once the expert has placed its gripper near
the cube, its state already carries useful location information. Initial-frame
diagnostics make that difference visible without giving object poses to the model.

## Interpreting the numbers

- Low training loss with poor rollout success can indicate compounding error,
  insufficient recovery demonstrations, task imbalance, or an unsuitable action
  distribution.
- High validation loss with lower training loss can suggest overfitting, but flow
  loss is stochastic and depends on sampled noise/time. Use fixed seeds and enough
  samples, and inspect the physical policy as well.
- Loss may improve before a single complete pick-and-place succeeds. Completion
  requires a sequence of approach, contact, transport, release, and retreat.
- Checkpoint existence and CUDA usage prove that the pipeline executed; neither
  proves language grounding or generalization.

Reference: [official SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla)
and [pinned trainer source](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_train.py).
