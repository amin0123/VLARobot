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
| `evaluation_results.csv` | `outputs/evaluation/results.csv` |
| `evaluation_summary.json` | `outputs/evaluation/summary.json` |
| `evaluation_report.md` | `outputs/evaluation/report.md` |
| `evaluation_run_config.json` | `outputs/evaluation/run_config.json` |
| `evaluation_failure_analysis.json` | `outputs/evaluation/failure_analysis.json` |
| `evaluation_failure_analysis.md` | `outputs/evaluation/failure_analysis.md` |
| `evaluation_artifact_integrity.json` | `outputs/evaluation/artifact_integrity.json` |
| `gui_synchronization.json` | `outputs/checks/gui_synchronization/simulation_check.json` |
| `gui_policy_result.json` | Result section of `outputs/videos/interactive_20261004T164634_355991Z.json` |

The GUI result file extracts the result section; all other JSON/CSV report copies
are unchanged. The parent [experiment log](../experiments.md) distinguishes the scripted teacher,
offline action diagnostics, training loss, and physical learned-policy success.
Absolute paths inside runtime metadata describe the machine used for the run.
The original local artifacts remain under `outputs/` and `datasets/`.
