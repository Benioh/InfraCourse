"""
L36 Patch · WeightSyncCoordinator

填空规则：
- TODO(student) 必须自己写
- 不许 import torch.distributed（本关单机模拟）
- 允许 torch / dict 操作

完成度自检：
    make patch-test M=l32_slime_rl_core
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch


class WeightSyncCoordinator:
    """从 train 侧拉 state_dict，按 shape + dtype 匹配后写入 inference 侧。"""

    def __init__(
        self,
        train_state_provider: Callable[[], Dict[str, torch.Tensor]],
        inference_state_setter: Callable[[Dict[str, torch.Tensor]], None],
        inference_state_provider: Callable[[], Dict[str, torch.Tensor]] | None = None,
    ) -> None:
        # TODO(student):
        #   self.train_provider = train_state_provider
        #   self.inference_setter = inference_state_setter
        #   self.inference_provider = inference_state_provider
        raise NotImplementedError("L36: implement __init__")

    def sync(self) -> Dict[str, object]:
        """返回 {bytes_synced, num_tensors, mismatched_keys}.

        规则：
          - 取 train_state 与 inference_state（如果提供 provider）
          - 对每个 train key：
              - 如果不在 inference state → 跳过，加入 mismatched_keys
              - 如果 shape / dtype 不匹配 → 跳过，加入 mismatched_keys
              - 否则：复制 tensor 到 inference 的 dict 里，累加 bytes
          - 调 inference_state_setter(new_state)
        """
        # TODO(student):
        #   train_state = self.train_provider()
        #   if self.inference_provider is None:
        #       # 极简模式：全 copy（假设 setter 会自己校验）
        #       self.inference_setter(dict(train_state))
        #       total = sum(t.numel() * t.element_size() for t in train_state.values())
        #       return {"bytes_synced": total, "num_tensors": len(train_state), "mismatched_keys": []}
        #
        #   inference_state = self.inference_provider()
        #   accepted = {}
        #   mismatched: List[str] = []
        #   bytes_synced = 0
        #
        #   for k, v in train_state.items():
        #       if k not in inference_state:
        #           mismatched.append(k); continue
        #       inf_v = inference_state[k]
        #       if v.shape != inf_v.shape or v.dtype != inf_v.dtype:
        #           mismatched.append(k); continue
        #       accepted[k] = v.detach().clone()
        #       bytes_synced += v.numel() * v.element_size()
        #
        #   self.inference_setter(accepted)
        #   return {
        #       "bytes_synced": bytes_synced,
        #       "num_tensors": len(accepted),
        #       "mismatched_keys": mismatched,
        #   }
        raise NotImplementedError("L36: implement sync")
