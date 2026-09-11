from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    train_widths = [float(x) for x in config["compression"]["train_widths"]]
    eval_widths = [float(x) for x in config["compression"]["eval_widths"]]
    if train_widths != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError("Anchor-only baseline requires train_widths=[0.25, 0.5, 0.75, 1.0]")
    missing = set(train_widths) - set(eval_widths)
    if missing:
        raise ValueError(f"Training widths missing from evaluation grid: {sorted(missing)}")
    if eval_widths != sorted(set(eval_widths)):
        raise ValueError("eval_widths must be sorted and unique")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def budget_tag(width: float) -> str:
    return f"{int(round(width * 100)):03d}"


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)
