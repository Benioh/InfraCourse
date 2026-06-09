"""
L04 Patch · 手写 ManualDDP

填空规则：
- 凡是 `# TODO(student): ...` 必须自己写。
- 不许 import torch.nn.parallel.* 或 torch.distributed.fsdp.*。
- 允许 torch.distributed.all_reduce 等低层 API。

完成度自检：
    make patch-test M=l03_nccl_ddp_smoke
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


class ManualDDP:
    """简化版 DDP：构造时不挂 hook，只在 synchronize_grads() 时同步。

    真实的 PyTorch DDP 在 backward 进行中就 bucketed all-reduce 重叠通信，
    我们这里先不做 overlap——把通信全堆到 backward 之后，便于读懂数学。
    """

    def __init__(
        self,
        model: nn.Module,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        # TODO(student):
        #   1. self.module = model（保留原 model，不需要拷贝）
        #   2. self.process_group = process_group
        #   3. self.world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        #
        # 注意：不要把 param 复制一份，self.module.parameters() 必须就是 model.parameters() 同一份内存。
        raise NotImplementedError("L04 Patch: implement ManualDDP.__init__")

    def synchronize_grads(self) -> None:
        """In-place 把所有 param.grad 从 local 改成 (sum across ranks) / world_size。

        关键决策（任一种都行）：
        - 顺序 all-reduce：循环每个 param，dist.all_reduce(p.grad)
        - coalesced all-reduce：调 dist.all_reduce_coalesced 一次同步全部 grad
        - 也可以先 grad /= world_size 再 all_reduce（等价）

        本关只看结果对不对，不规定方式。
        """
        # TODO(student):
        #   if self.world_size == 1 or not dist.is_initialized(): return
        #
        #   遍历 self.module.parameters()，对每个 p:
        #     if p.requires_grad and p.grad is not None:
        #       dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.process_group)
        #       p.grad /= self.world_size
        raise NotImplementedError("L04 Patch: implement ManualDDP.synchronize_grads")
