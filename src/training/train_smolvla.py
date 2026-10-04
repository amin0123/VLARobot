"""Thin runner around official lerobot-train, with local logs and bounded OOM retries."""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.common import ROOT, config, write_json


def gpu_monitor(path, stop):
    with path.open("w") as stream:
        stream.write("wall_time,memory_used_mib,utilization_percent\n")
        while not stop.is_set():
            try:
                value = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    )
                    .strip()
                    .splitlines()[0]
                )
                stream.write(f"{time.time()},{value}\n")
                stream.flush()
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
            stop.wait(5)


def command(cfg, output, batch_size):
    # Loading SmolVLA's full pretrained state avoids a redundant VLM weight load.
    flags = {
        "policy.path": str(Path(cfg["checkpoint"]).resolve())
        if Path(cfg["checkpoint"]).exists()
        else cfg["checkpoint"],
        "policy.device": cfg["device"],
        "policy.use_amp": str(cfg["mixed_precision"]).lower(),
        "policy.push_to_hub": "false",
        "policy.load_vlm_weights": "false",
        "policy.input_features": "null",
        "policy.output_features": "null",
        "policy.resize_imgs_with_padding": json.dumps([cfg["image_size"]] * 2),
        "policy.chunk_size": cfg["chunk_size"],
        "policy.n_action_steps": cfg["n_action_steps"],
        "policy.optimizer_lr": cfg["learning_rate"],
        "policy.scheduler_warmup_steps": cfg["warmup_steps"],
        "policy.scheduler_decay_steps": cfg["steps"],
        "dataset.repo_id": cfg["repo_id"],
        "dataset.root": str(Path(cfg["dataset_root"]).resolve()),
        "dataset.video_backend": "pyav",
        "dataset.eval_split": 0,
        "output_dir": str(output.resolve()),
        "job_name": "vlarobot",
        "batch_size": batch_size,
        "steps": cfg["steps"],
        "save_freq": cfg["save_freq"],
        "log_freq": cfg["log_freq"],
        "env_eval_freq": 0,
        "eval_steps": 0,
        "num_workers": cfg["num_workers"],
        "seed": cfg["seed"],
        "wandb.enable": "false",
    }
    return [sys.executable, "-m", "src.training.official_entrypoint"] + [
        f"--{k}={v}" for k, v in flags.items()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    for attr, key in (
        ("steps", "steps"),
        ("batch_size", "batch_size"),
        ("output", "output_dir"),
        ("checkpoint", "checkpoint"),
    ):
        if getattr(args, attr) is not None:
            cfg[key] = getattr(args, attr)
    if cfg["steps"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("steps and batch size must be positive")
    cfg["warmup_steps"] = min(cfg["warmup_steps"], max(1, cfg["steps"] // 10))
    output = Path(cfg["output_dir"])
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose --output to preserve this experiment.")
    if not (Path(cfg["dataset_root"]) / "meta/info.json").is_file():
        raise RuntimeError("Collect, inspect, and split the dataset before training.")
    if not (ROOT / "outputs/smoke_test_vla.json").is_file():
        raise RuntimeError("Run python scripts/smoke_test_vla.py before training.")
    output.mkdir(parents=True)
    write_json(output / "requested_config.json", cfg)
    batch_size = cfg["batch_size"]
    start = time.monotonic()
    attempt = 0
    while batch_size >= 1:
        run_dir = output / f"attempt_{attempt:02d}_batch_{batch_size}"
        cmd = command(cfg, run_dir, batch_size)
        write_json(output / f"command_{attempt:02d}.json", cmd)
        stop = threading.Event()
        monitor = threading.Thread(
            target=gpu_monitor, args=(output / f"gpu_{attempt:02d}.csv", stop), daemon=True
        )
        monitor.start()
        tail, losses = [], []
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"})
        try:
            with (output / f"training_{attempt:02d}.log").open("w") as log:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                        tail = (tail + [line])[-100:]
                        match = re.search(r"\bloss:\s*([\d.eE+\-]+)", line)
                        if match:
                            losses.append(
                                {"wall_seconds": time.monotonic() - start, "loss": float(match.group(1))}
                            )
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
        finally:
            stop.set()
            monitor.join(timeout=6)
        write_json(output / f"losses_{attempt:02d}.json", losses)
        if returncode == 0:
            checkpoints = run_dir / "checkpoints"
            (output / "checkpoints").symlink_to(checkpoints.relative_to(output), target_is_directory=True)
            checkpoint = output / "checkpoints/last/pretrained_model"
            if not (checkpoint / "model.safetensors").is_file():
                raise RuntimeError("Trainer exited successfully without the expected model checkpoint.")
            report = {
                "status": "completed",
                "steps": cfg["steps"],
                "batch_size": batch_size,
                "oom_retries": attempt,
                "wall_seconds": time.monotonic() - start,
                "checkpoint": str(checkpoint),
                "loss_log_entries": len(losses),
                "initial_logged_loss": losses[0]["loss"] if losses else None,
                "final_logged_loss": losses[-1]["loss"] if losses else None,
                "experiment_scope": "sanity" if cfg["steps"] <= 500 else "longer_fine_tune",
            }
            write_json(output / "summary.json", report)
            if not args.skip_validation:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.training.validate",
                        "--checkpoint",
                        str(checkpoint),
                        "--root",
                        cfg["validation_root"],
                        "--samples",
                        str(cfg["max_eval_samples"]),
                        "--output",
                        str(output / "validation.json"),
                    ],
                    check=True,
                )
            print(json.dumps(report, indent=2))
            return
        text = "".join(tail).lower()
        oom = "out of memory" in text or "cuda error: out of memory" in text
        if not oom or batch_size == 1:
            raise RuntimeError(
                f"Training failed (exit {returncode}); see {output}/training_{attempt:02d}.log"
            )
        batch_size = max(1, batch_size // 2)
        attempt += 1
        print(
            f"CUDA OOM: restarting from pretrained weights with batch_size={batch_size}. Failed logs retained.",
            flush=True,
        )


if __name__ == "__main__":
    main()
