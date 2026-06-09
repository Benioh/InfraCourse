"""Reference solution for L09 Patch · CosineWithRestartsLR."""

from __future__ import annotations

import math
from typing import Sequence

import torch


class CosineWithRestartsLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: float,
        restart_steps: Sequence[int],
        total_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.restart_steps = list(restart_steps)
        self.total_steps = int(total_steps)
        self.step_count = 0
        self.boundaries = [0] + list(self.restart_steps) + [self.total_steps]
        self._set_lr(self.max_lr)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _compute_lr(self, step: int) -> float:
        if step >= self.total_steps:
            return self.min_lr
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= step < self.boundaries[i + 1]:
                segment_start = self.boundaries[i]
                segment_end = self.boundaries[i + 1]
                break
        else:
            return self.min_lr
        segment_len = segment_end - segment_start
        if segment_len <= 0:
            return self.max_lr
        t = step - segment_start
        ratio = t / segment_len
        cos_factor = 0.5 * (1 + math.cos(math.pi * ratio))
        return self.min_lr + (self.max_lr - self.min_lr) * cos_factor

    def step(self) -> None:
        self.step_count += 1
        self._set_lr(self._compute_lr(self.step_count))

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
