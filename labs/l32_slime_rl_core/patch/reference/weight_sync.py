"""Reference solution for L11 Patch · WeightSyncCoordinator."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch


class WeightSyncCoordinator:
    def __init__(
        self,
        train_state_provider: Callable[[], Dict[str, torch.Tensor]],
        inference_state_setter: Callable[[Dict[str, torch.Tensor]], None],
        inference_state_provider: Optional[Callable[[], Dict[str, torch.Tensor]]] = None,
    ) -> None:
        self.train_provider = train_state_provider
        self.inference_setter = inference_state_setter
        self.inference_provider = inference_state_provider

    def sync(self) -> Dict[str, object]:
        train_state = self.train_provider()

        if self.inference_provider is None:
            # 简化模式：全量推送
            new_state = {k: v.detach().clone() for k, v in train_state.items()}
            self.inference_setter(new_state)
            total = sum(t.numel() * t.element_size() for t in new_state.values())
            return {
                "bytes_synced": total,
                "num_tensors": len(new_state),
                "mismatched_keys": [],
            }

        inference_state = self.inference_provider()
        accepted: Dict[str, torch.Tensor] = {}
        mismatched: List[str] = []
        bytes_synced = 0

        for k, v in train_state.items():
            if k not in inference_state:
                mismatched.append(k)
                continue
            inf_v = inference_state[k]
            if tuple(v.shape) != tuple(inf_v.shape) or v.dtype != inf_v.dtype:
                mismatched.append(k)
                continue
            accepted[k] = v.detach().clone()
            bytes_synced += v.numel() * v.element_size()

        self.inference_setter(accepted)
        return {
            "bytes_synced": bytes_synced,
            "num_tensors": len(accepted),
            "mismatched_keys": mismatched,
        }
