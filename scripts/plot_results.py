"""Plot measured local results; never supply synthetic curves or success rates."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, nargs="*", default=[])
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.training:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for root in args.training:
            summary = json.loads((root / "summary.json").read_text())
            points = json.loads((root / f"losses_{summary['oom_retries']:02d}.json").read_text())
            if not points:
                raise ValueError(f"No measured loss entries in {root}")
            ax.plot([p["wall_seconds"] / 60 for p in points], [p["loss"] for p in points], label=root.name)
        ax.set(
            xlabel="Training wall time (minutes)",
            ylabel="Logged flow-matching loss",
            title="SmolVLA fine-tuning",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(args.output / "training_loss.png", dpi=180)
        plt.close(fig)
    if args.evaluation:
        report = json.loads((args.evaluation / "summary.json").read_text())
        names, stats = zip(*report["conditions"].items(), strict=True)
        rates = np.array([s["success_rate"] for s in stats]) * 100
        errors = np.array(
            [
                [r - s["wilson_95_low"] * 100 for r, s in zip(rates, stats, strict=True)],
                [s["wilson_95_high"] * 100 - r for r, s in zip(rates, stats, strict=True)],
            ]
        )
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        bars = ax.bar(names, rates, yerr=np.maximum(errors, 0), capsize=4, color="#276b9e")
        ax.bar_label(bars, labels=[f"{s['successes']}/{s['episodes']}" for s in stats], padding=6)
        ax.set(ylim=(0, 110), ylabel="Task success (%)", title="Closed-loop evaluation: 95% Wilson intervals")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(args.output / "evaluation.png", dpi=180)
        plt.close(fig)
    print(f"Saved measured plots under {args.output}")


if __name__ == "__main__":
    main()
