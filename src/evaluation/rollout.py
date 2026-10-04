"""One closed-loop rollout. Privileged telemetry is used after action prediction."""

import time

from src.common import IMAGE_KEY, config
from src.evaluation.metrics import classify_failure
from src.video import VideoRecorder


def rollout(sim, policy, task, seed, condition="position", max_steps=None, video_path=None, episode=0):
    c = config()
    obs, info = sim.reset(seed, task, condition=condition)
    policy.reset(seed=seed)
    trace, actions, latencies = [], [], []
    start = time.monotonic()
    video = VideoRecorder(video_path, c["control_hz"]) if video_path else None
    try:
        for step in range(min(max_steps or c["max_steps"], c["max_steps"])):
            # Only these sensors and the unmodified instruction reach SmolVLA.
            action = policy.act(obs, task.instruction)
            actions.append(action.tolist())
            latencies.append(policy.last_latency_s)
            trace.append(sim.call("privileged"))
            if video:
                video.add(
                    obs[IMAGE_KEY],
                    task.instruction,
                    f"VLA | {condition} | Episode {episode} | Step {step} | {policy.last_latency_s * 1000:.0f} ms",
                )
            obs, info = sim.step(action)
            if info["terminated"] or info["truncated"]:
                break
        trace.append(sim.call("privileged"))
        sim.call("status", text="SUCCESS" if info["success"] else "FAILURE / TIMEOUT")
        if video:
            for _ in range(10):
                video.add(
                    obs[IMAGE_KEY],
                    task.instruction,
                    f"VLA | {'SUCCESS' if info['success'] else 'FAILURE'} | {info['step']} steps",
                )
    finally:
        if video:
            video.close()
    row = {
        "condition": condition,
        "episode": episode,
        "seed": seed,
        "policy_seed": seed,
        **task.to_dict(),
        "success": bool(info["success"]),
        "episode_length": info["step"],
        "completion_time_s": info["sim_seconds"] if info["success"] else None,
        "sim_duration_s": info["sim_seconds"],
        "wall_duration_s": time.monotonic() - start,
        "inference_wall_s": sum(latencies),
        "video": str(video_path) if video_path else "",
        "failure_category": "" if info["success"] else classify_failure(trace, task.to_dict(), c),
        "initial_configuration": info["initial_configuration"],
    }
    return row, {"task": task.to_dict(), "trace": trace, "actions": actions, "latency_s": latencies}
