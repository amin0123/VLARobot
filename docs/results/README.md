# Recorded experiment evidence

These small files are exact copies of measurements produced on 2026-10-04.
They are checked into Git so the educational record survives without committing
large datasets, optimizer states, model weights, or videos.

| File | Original artifact |
|---|---|
| `expert_summary.json` | `outputs/expert/summary.json` |
| `dataset_inspection.json` | `datasets/pick_place_50/meta/inspection.json` |
| `full_training_summary.json` | `outputs/train/full/summary.json` |
| `full_training_loss.json` | `outputs/train/full/losses_00.json` |
| `full_validation.json` | `outputs/train/full/validation.json` |
| `full_episode_start_diagnostics.json` | `outputs/action_diagnostics_full_starts.json` |
| `pinned_model_equivalence.json` | `outputs/checks/pinned_model_equivalence.json` |
| `smoke_test_vla_pinned.json` | `outputs/smoke_test_vla_pinned.json` |

The parent [experiment log](../experiments.md) distinguishes the scripted teacher,
offline action diagnostics, training loss, and physical learned-policy success.
Absolute paths inside runtime metadata describe the machine used for the run.
The original local artifacts remain under `outputs/` and `datasets/`.
