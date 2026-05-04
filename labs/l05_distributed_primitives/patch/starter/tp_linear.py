"""
L02 Patch · 手写 Tensor Parallel Linear
========================================

填空规则：
- 凡是 `# TODO(student): ...` 必须自己写。
- 不许 import megatron.* 或 torch.nn.parallel.DistributedDataParallel。
- 允许 import torch.distributed as dist 并使用 all_reduce / all_gather。
- 保持类签名，不许改 __init__ 参数名（test 会按这些名字构造）。

完成度自检：
    make patch-test M=l05_distributed_primitives
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


# ---------------------------------------------------------------------------
# autograd.Function 通信 primitive：你需要补全 forward 和 backward
# ---------------------------------------------------------------------------


class _CopyToParallelRegion(torch.autograd.Function):
    """forward: identity；backward: all-reduce(SUM) over process group."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
        ctx.group = group
        # TODO(student): 1 行 — 直接返回 x（identity）
        raise NotImplementedError("L02 Patch: implement _CopyToParallelRegion.forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO(student): 在 ctx.group 上对 grad_output 做 all-reduce(SUM)，再返回 (grad_output, None)
        raise NotImplementedError("L02 Patch: implement _CopyToParallelRegion.backward")


class _ReduceFromParallelRegion(torch.autograd.Function):
    """forward: all-reduce(SUM)；backward: identity."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
        ctx.group = group
        # TODO(student): 对 x 做 all-reduce(SUM) inplace 然后返回 x
        raise NotImplementedError("L02 Patch: implement _ReduceFromParallelRegion.forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO(student): backward 是 identity；返回 (grad_output, None)
        raise NotImplementedError("L02 Patch: implement _ReduceFromParallelRegion.backward")


class _GatherAlongLastDim(torch.autograd.Function):
    """forward: all-gather along last dim；backward: split."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        ctx.world_size = world_size
        ctx.group = group
        if world_size == 1:
            return x
        # TODO(student):
        #   1. 在最后一维 all-gather x 的所有分片
        #   2. 把 list[tensor] 拼成一个 tensor 返回
        raise NotImplementedError("L02 Patch: implement _GatherAlongLastDim.forward")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.world_size == 1:
            return grad_output, None
        # TODO(student): 把 grad_output 沿最后一维切成 world_size 份，返回当前 rank 那一份
        raise NotImplementedError("L02 Patch: implement _GatherAlongLastDim.backward")


# ---------------------------------------------------------------------------
# 主类：ColumnParallelLinear / RowParallelLinear
# ---------------------------------------------------------------------------


class ColumnParallelLinear(nn.Module):
    """
    把 weight (out_features, in_features) 沿 out_features 切成 world_size 份。

    forward:
        x : (..., in_features)            (全卡相同)
        out_local = x @ W_local^T + b_local
        if gather_output:
            return all_gather(out_local) along last dim   -> (..., out_features)
        else:
            return out_local                              -> (..., out_features // ws)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        gather_output: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        device=None,
        dtype=None,
    ) -> None:
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
        # TODO(student):
        #   1. 创建本地权重 self.weight，shape = (out_per_partition, in_features)
        #   2. 用 kaiming_uniform_(a=sqrt(5)) 初始化（与 nn.Linear 一致）
        #   3. 如果 bias，创建本地 bias self.bias，shape = (out_per_partition,)，初始化为 uniform(-bound, bound)，bound = 1/sqrt(in_features)
        #   4. 否则 self.register_parameter("bias", None)
        raise NotImplementedError("L02 Patch: implement ColumnParallelLinear.__init__ params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(student):
        #   1. x = _CopyToParallelRegion.apply(x, self.process_group)
        #      （forward 为 identity；backward 时把 grad-input all-reduce）
        #   2. 本地 matmul：out_local = F.linear(x, self.weight, self.bias)
        #   3. if self.gather_output: out = _GatherAlongLastDim.apply(out_local, self.process_group)
        #      else:                   out = out_local
        #   4. return out
        raise NotImplementedError("L02 Patch: implement ColumnParallelLinear.forward")


class RowParallelLinear(nn.Module):
    """
    把 weight (out_features, in_features) 沿 in_features 切成 world_size 份。

    forward:
        if input_is_parallel:
            x : (..., in_features // ws)   (各卡分片)
        else:
            x : (..., in_features)         (全卡相同, 需要本地切片)
        out_local = x_local @ W_local^T
        out = all_reduce(out_local)
        if rank == 0 and bias is not None:
            out = out + bias
        return out                         -> (..., out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        input_is_parallel: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        device=None,
        dtype=None,
    ) -> None:
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
        # TODO(student):
        #   1. self.weight：shape = (out_features, in_per_partition)，kaiming_uniform_
        #   2. 如果 bias：创建 self.bias，shape = (out_features,) —— 注意是完整长度！
        #      bias 是 **replicated** 的（每张卡持有相同副本），不是切分的。
        #      用 uniform_(-bound, bound)，bound = 1/sqrt(in_features) 初始化，
        #      然后 dist.broadcast(self.bias.data, src=0) 保证各 rank 一致。
        #   3. 否则 self.register_parameter("bias", None)
        #
        # 为什么 bias 要 replicate 不切分？因为在 forward 里 bias 必须在 all-reduce
        # **之后** 加，且只加一次；如果 bias 切分了你就得自己拼回来——多此一举。
        raise NotImplementedError("L02 Patch: implement RowParallelLinear.__init__ params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(student):
        #   1. if not self.input_is_parallel:
        #          沿最后一维把 x 切成 world_size 份，取本 rank 的那份 -> x_local
        #      else:
        #          x_local = x
        #   2. out_local = F.linear(x_local, self.weight)   # 注意：这里不加 bias！
        #   3. out = _ReduceFromParallelRegion.apply(out_local, self.process_group)
        #   4. if self.bias is not None: out = out + self.bias   # all-reduce 之后才加
        #
        #   关键陷阱：如果你在第 2 步用 F.linear(x_local, self.weight, self.bias)，
        #   bias 会被加在每个 rank 的 local 输出上，all-reduce 求和后变成 bias × world_size。
        #   test_row_bias_added_once 就是来抓这个 bug 的。
        raise NotImplementedError("L02 Patch: implement RowParallelLinear.forward")
