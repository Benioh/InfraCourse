"""
L09 Patch · Cosine With Restarts LR Scheduler

填空规则：
- TODO(student) 必须自己写
- 不许用 torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
- 允许 math / torch.optim.Optimizer.param_groups

完成度自检：
    make patch-test M=l08_megatron_text_pretrain
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


class CosineWithRestartsLR:
    """带热重启的 cosine LR scheduler。

    在 restart_steps 之间做余弦衰减，每个 restart_step 时刻把 lr 重置回 max_lr。

    Example:
        >>> sched = CosineWithRestartsLR(opt, max_lr=1e-3, min_lr=1e-5,
        ...     restart_steps=[1000, 3000], total_steps=5000)
        >>> for s in range(5000):
        ...     opt.step(); sched.step()
    """

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

        # 把每段的边界算出来：[0, r1, r2, ..., total_steps]
        # TODO(student): 计算 self.boundaries 列表，长度 = len(restart_steps) + 2
        #   例：restart_steps=[1000, 3000], total_steps=5000 → boundaries = [0, 1000, 3000, 5000]
        raise NotImplementedError("L09 Patch: implement boundaries init")

        # 初始 lr
        self._set_lr(self.max_lr)

    def _set_lr(self, lr: float) -> None:
        # TODO(student): 把 lr 写入 self.optimizer.param_groups 中所有 group 的 'lr'。
        raise NotImplementedError("L09 Patch: implement _set_lr")

    def _compute_lr(self, step: int) -> float:
        if step >= self.total_steps:
            return self.min_lr
        # TODO(student):
        #   1. 找到 step 落在哪一段：
        #      for i in range(len(self.boundaries) - 1):
        #          if self.boundaries[i] <= step < self.boundaries[i + 1]:
        #              segment_start = self.boundaries[i]
        #              segment_end = self.boundaries[i + 1]
        #              break
        #   2. 段内相对位置 t = step - segment_start
        #   3. segment_len = segment_end - segment_start
        #   4. ratio = t / segment_len
        #   5. cos_factor = 0.5 * (1 + math.cos(math.pi * ratio))
        #   6. return self.min_lr + (self.max_lr - self.min_lr) * cos_factor
        raise NotImplementedError("L09 Patch: implement _compute_lr")

    def step(self) -> None:
        """每个 optimizer.step() 之后调用一次。"""
        self.step_count += 1
        new_lr = self._compute_lr(self.step_count)
        self._set_lr(new_lr)

    def get_lr(self) -> float:
        """返回当前 lr（与 optimizer.param_groups[0]['lr'] 一致）。"""
        return self.optimizer.param_groups[0]["lr"]
