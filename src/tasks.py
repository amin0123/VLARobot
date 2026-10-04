"""Task metadata and splits. This module never supplies coordinates to the VLA."""

import re
from dataclasses import asdict, dataclass

import numpy as np

from src.common import COLORS

TRAIN_TEMPLATES = (
    "Put the {source} cube in the {destination} box.",
    "Move the {source} cube into the {destination} box.",
    "Place the {source} block inside the {destination} container.",
    "Pick up the {source} object and put it in the {destination} container.",
    "Take the {source} cube to the {destination} box.",
)
TEST_TEMPLATES = (
    "Please transfer the {source} cube into the {destination} container.",
    "The {source} block belongs in the {destination} box; place it there.",
    "Put the {source} object inside the container colored {destination}.",
)
HELD_OUT_PAIRS = (("green", "blue"), ("blue", "yellow"), ("yellow", "red"))
TRAIN_PAIRS = tuple((a, b) for a in COLORS for b in COLORS if a != b and (a, b) not in HELD_OUT_PAIRS)


@dataclass(frozen=True)
class Task:
    source: str
    destination: str
    instruction: str
    template_id: str = "user"

    def __post_init__(self):
        if self.source not in COLORS or self.destination not in COLORS:
            raise ValueError("Source and destination must be red, green, blue, or yellow.")

    def to_dict(self):
        return asdict(self)


def sample_task(index: int, seed: int, condition="train") -> Task:
    """Balanced task pairs; deterministic independently of scene placement RNG."""
    pairs = HELD_OUT_PAIRS if condition == "composition" else TRAIN_PAIRS
    source, destination = pairs[index % len(pairs)]
    templates = TEST_TEMPLATES if condition == "language" else TRAIN_TEMPLATES
    rng = np.random.default_rng(np.random.SeedSequence([seed, 731]))
    template_idx = int(rng.integers(len(templates)))
    return Task(
        source,
        destination,
        templates[template_idx].format(source=source, destination=destination),
        f"{'test' if condition == 'language' else 'train'}_{template_idx}",
    )


def evaluation_task(index, seed, condition, training_rows, instruction_sampling="seen"):
    """Hold actual recorded wording fixed across the position/visual conditions."""
    task = sample_task(index, seed, condition)
    if instruction_sampling == "seen" and condition in ("position", "lighting", "camera", "scale"):
        for row in training_rows:
            if (row["source"], row["destination"]) == (task.source, task.destination):
                return Task(row["source"], row["destination"], row["instruction"], row["template_id"])
        raise ValueError(f"No training instruction for {task.source} → {task.destination}")
    return task


def task_from_instruction(instruction: str, source=None, destination=None) -> Task:
    """Resolve *metric labels only*. The original instruction goes unchanged to SmolVLA.

    This controlled vocabulary parser is not a language planner or policy input.
    Ambiguous commands require explicit metric labels rather than silently guessing.
    """
    if source is None or destination is None:
        colors = re.findall(r"\b(red|green|blue|yellow)\b", instruction.lower())
        if len(colors) != 2:
            raise ValueError("For scoring, use two color names or supply --source and --destination.")
        source, destination = colors
    return Task(source, destination, instruction)
