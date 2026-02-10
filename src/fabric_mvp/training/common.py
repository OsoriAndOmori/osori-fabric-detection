from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EarlyStopping:
    patience: int = 8
    min_delta: float = 1e-4
    best: float = float("-inf")
    counter: int = 0

    def step(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(path: Path, model: torch.nn.Module, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_state": model.state_dict(), **payload}
    torch.save(state, str(path))
