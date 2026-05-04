"""
L10 Patch · Adaptive KL Controller (Ouyang 2022)

填空规则：
- TODO(student) 必须自己写
- 不许 import trl / verl
- 允许 stdlib

完成度自检：
    make patch-test M=l29_verl_rl_baseline
"""

from __future__ import annotations


class AdaptiveKLController:
    """根据 current_kl 与 target_kl 的偏差自动调节 kl_coef.

    遵循 Ouyang 2022 InstructGPT App C 公式：
        proportional_error = clip(current_kl / target_kl - 1, -0.2, 0.2)
        kl_coef *= 1 + proportional_error * n_steps / horizon
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int) -> None:
        # TODO(student):
        #   self.value = float(init_kl_coef)
        #   self.target_kl = float(target_kl)
        #   self.horizon = int(horizon)
        raise NotImplementedError("L10: implement __init__")

    def update(self, current_kl: float, n_steps: int = 1) -> None:
        """根据 current_kl 与 target_kl 的偏差更新 self.value."""
        # TODO(student):
        #   error = current_kl / self.target_kl - 1.0
        #   error = max(-0.2, min(0.2, error))   # clip
        #   self.value *= 1.0 + error * n_steps / self.horizon
        raise NotImplementedError("L10: implement update")

    def get_coef(self) -> float:
        # TODO(student): return self.value
        raise NotImplementedError("L10: implement get_coef")
