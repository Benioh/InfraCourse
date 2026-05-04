"""Reference solution for L10 Patch · AdaptiveKLController."""

from __future__ import annotations


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int) -> None:
        self.value = float(init_kl_coef)
        self.target_kl = float(target_kl)
        self.horizon = int(horizon)

    def update(self, current_kl: float, n_steps: int = 1) -> None:
        error = current_kl / self.target_kl - 1.0
        error = max(-0.2, min(0.2, error))
        self.value *= 1.0 + error * n_steps / self.horizon

    def get_coef(self) -> float:
        return self.value
