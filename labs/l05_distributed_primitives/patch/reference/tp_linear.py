"""Reference solution for L06 Patch · Tensor Parallel Linear.

This file is the canonical answer. Tests can import from either
`starter.tp_linear` or `reference.tp_linear` via the
TP_IMPL=starter|reference env var, so we can run the test harness
against the reference to confirm the harness itself is correct.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class _CopyToParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized() and dist.get_world_size(ctx.group) > 1:
            grad_output = grad_output.contiguous()
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        if dist.is_initialized() and dist.get_world_size(group) > 1:
            x = x.contiguous()
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _GatherAlongLastDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        ctx.world_size = world_size
        ctx.group = group
        if world_size == 1:
            return x
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.world_size == 1:
            return grad_output, None
        rank = dist.get_rank(ctx.group)
        chunks = grad_output.chunk(ctx.world_size, dim=-1)
        return chunks[rank].contiguous(), None


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        bias=True,
        gather_output=True,
        process_group=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.process_group = process_group

        world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        if out_features % world_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by world_size ({world_size})"
            )
        self.out_per_partition = out_features // world_size
        self.world_size = world_size

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(self.out_per_partition, in_features, **factory_kwargs)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_partition, **factory_kwargs))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = _CopyToParallelRegion.apply(x, self.process_group)
        out_local = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            return _GatherAlongLastDim.apply(out_local, self.process_group)
        return out_local


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        bias=True,
        input_is_parallel=False,
        process_group=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.process_group = process_group

        world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        if in_features % world_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by world_size ({world_size})"
            )
        self.in_per_partition = in_features // world_size
        self.world_size = world_size
        self.rank = dist.get_rank(process_group) if dist.is_initialized() else 0

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_per_partition, **factory_kwargs)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            # bias 是 *replicated* 的：每张卡持有同样的值，
            # all-reduce 之后只加一次。这就是 Megatron 的做法。
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
            if dist.is_initialized() and world_size > 1:
                # 各 rank 的随机数可能不同步，做一次 broadcast 保证一致。
                dist.broadcast(self.bias.data, src=0, group=process_group)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if not self.input_is_parallel:
            chunks = x.chunk(self.world_size, dim=-1)
            x_local = chunks[self.rank].contiguous()
        else:
            x_local = x
        # 关键：bias 必须在 all-reduce **之后** 加，否则会被加 world_size 次。
        out_local = F.linear(x_local, self.weight)  # no bias yet
        out = _ReduceFromParallelRegion.apply(out_local, self.process_group)
        if self.bias is not None:
            out = out + self.bias
        return out
