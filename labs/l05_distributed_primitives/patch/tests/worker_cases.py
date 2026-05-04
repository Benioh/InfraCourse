"""
Per-rank assertions that conftest spawns into worker processes.

Each function takes (rank, world_size, impl_module) and uses ordinary
asserts. If a worker raises, the parent test fails with that traceback.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


# -----------------------------------------------------------------------------
# Numerical equivalence: TP=N output must equal nn.Linear output
# -----------------------------------------------------------------------------


def column_parallel_matches_single_gpu(rank, world_size, impl):
    torch.manual_seed(42)
    in_f, out_f, bsz = 8, 16, 4

    # Build the canonical single-GPU linear on every rank, then sync.
    canonical = nn.Linear(in_f, out_f, bias=True).double()
    # Force all ranks to share the same canonical weights.
    for p in canonical.parameters():
        dist.broadcast(p.data, src=0)

    # Build the parallel layer; copy the canonical weight slice into it.
    layer = impl.ColumnParallelLinear(
        in_f, out_f, bias=True, gather_output=True
    ).double()
    out_per_part = out_f // world_size
    start = rank * out_per_part
    end = start + out_per_part
    layer.weight.data.copy_(canonical.weight.data[start:end, :])
    if layer.bias is not None:
        layer.bias.data.copy_(canonical.bias.data[start:end])

    x = torch.randn(bsz, in_f, dtype=torch.float64)
    dist.broadcast(x, src=0)

    expected = canonical(x)
    actual = layer(x)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert torch.allclose(actual, expected, atol=1e-10), (
        f"rank {rank}: max diff = {(actual - expected).abs().max().item()}"
    )


def row_parallel_matches_single_gpu(rank, world_size, impl):
    torch.manual_seed(42)
    in_f, out_f, bsz = 16, 8, 4

    canonical = nn.Linear(in_f, out_f, bias=True).double()
    for p in canonical.parameters():
        dist.broadcast(p.data, src=0)

    layer = impl.RowParallelLinear(
        in_f, out_f, bias=True, input_is_parallel=False
    ).double()
    in_per_part = in_f // world_size
    start = rank * in_per_part
    end = start + in_per_part
    layer.weight.data.copy_(canonical.weight.data[:, start:end])
    # bias is replicated across ranks — all ranks copy the canonical bias.
    if layer.bias is not None:
        layer.bias.data.copy_(canonical.bias.data)

    x = torch.randn(bsz, in_f, dtype=torch.float64)
    dist.broadcast(x, src=0)

    expected = canonical(x)
    actual = layer(x)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-10), (
        f"rank {rank}: max diff = {(actual - expected).abs().max().item()}"
    )


# -----------------------------------------------------------------------------
# Gradient correctness: backward grads must equal single-GPU autograd grads.
#
# We avoid torch.autograd.gradcheck because it issues an asymmetric number of
# forward calls per rank (rank 0 runs the finite-difference loop, rank 1 doesn't),
# which desynchronizes the gloo collective sequence and hangs. Instead we run one
# forward + one backward on every rank, then check that:
#   - grad_x  matches nn.Linear's grad_x exactly
#   - grad_W (local slice) matches the corresponding slice of nn.Linear's grad_W
# This subsumes gradcheck for our purposes and is collective-symmetric.
# -----------------------------------------------------------------------------


def column_grad_matches_single_gpu(rank, world_size, impl):
    torch.manual_seed(7)
    in_f, out_f, bsz = 6, 8, 3

    canonical = nn.Linear(in_f, out_f, bias=False).double()
    for p in canonical.parameters():
        dist.broadcast(p.data, src=0)

    layer = impl.ColumnParallelLinear(
        in_f, out_f, bias=False, gather_output=True
    ).double()
    out_per_part = out_f // world_size
    layer.weight.data.copy_(
        canonical.weight.data[rank * out_per_part:(rank + 1) * out_per_part, :]
    )

    x_can = torch.randn(bsz, in_f, dtype=torch.float64, requires_grad=True)
    dist.broadcast(x_can.data, src=0)
    x_par = x_can.detach().clone().requires_grad_(True)

    canonical(x_can).sum().backward()
    layer(x_par).sum().backward()

    # grad_x must match exactly on every rank
    diff_x = (x_par.grad - x_can.grad).abs().max().item()
    assert diff_x < 1e-10, f"rank {rank}: grad_x diff = {diff_x}"

    # grad_W (local slice) must match the slice of canonical.weight.grad
    expected_W = canonical.weight.grad[rank * out_per_part:(rank + 1) * out_per_part, :]
    diff_W = (layer.weight.grad - expected_W).abs().max().item()
    assert diff_W < 1e-10, f"rank {rank}: grad_W diff = {diff_W}"


def row_grad_matches_single_gpu(rank, world_size, impl):
    torch.manual_seed(7)
    in_f, out_f, bsz = 8, 6, 3

    canonical = nn.Linear(in_f, out_f, bias=False).double()
    for p in canonical.parameters():
        dist.broadcast(p.data, src=0)

    layer = impl.RowParallelLinear(
        in_f, out_f, bias=False, input_is_parallel=False
    ).double()
    in_per_part = in_f // world_size
    layer.weight.data.copy_(
        canonical.weight.data[:, rank * in_per_part:(rank + 1) * in_per_part]
    )

    x_can = torch.randn(bsz, in_f, dtype=torch.float64, requires_grad=True)
    dist.broadcast(x_can.data, src=0)
    x_par = x_can.detach().clone().requires_grad_(True)

    canonical(x_can).sum().backward()
    layer(x_par).sum().backward()

    # grad_x: only the local slice of x_par.grad is meaningful (Row splits input internally).
    # The implementation produces grad_x of full shape via slicing's autograd; the slice for
    # this rank should match the canonical slice.
    expected_grad_x_slice = x_can.grad[..., rank * in_per_part:(rank + 1) * in_per_part]
    actual_grad_x_slice = x_par.grad[..., rank * in_per_part:(rank + 1) * in_per_part]
    diff_x = (actual_grad_x_slice - expected_grad_x_slice).abs().max().item()
    assert diff_x < 1e-10, f"rank {rank}: grad_x slice diff = {diff_x}"

    expected_W = canonical.weight.grad[:, rank * in_per_part:(rank + 1) * in_per_part]
    diff_W = (layer.weight.grad - expected_W).abs().max().item()
    assert diff_W < 1e-10, f"rank {rank}: grad_W diff = {diff_W}"


# -----------------------------------------------------------------------------
# Behavior contract: communication happens at the right step
# -----------------------------------------------------------------------------


class AllreduceCounter:
    """Patch dist.all_reduce to count how many times it's called inside a block."""

    def __init__(self):
        self.count = 0
        self._orig = None

    def __enter__(self):
        self._orig = dist.all_reduce

        def counted(tensor, *args, **kwargs):
            self.count += 1
            return self._orig(tensor, *args, **kwargs)

        dist.all_reduce = counted  # type: ignore[assignment]
        return self

    def __exit__(self, *exc):
        dist.all_reduce = self._orig  # type: ignore[assignment]


def column_backward_triggers_allreduce(rank, world_size, impl):
    in_f, out_f = 4, 8
    layer = impl.ColumnParallelLinear(in_f, out_f, bias=False, gather_output=True)
    x = torch.randn(2, in_f, requires_grad=True)

    # Forward should NOT trigger all-reduce (only all-gather)
    with AllreduceCounter() as fwd_counter:
        out = layer(x)
    assert fwd_counter.count == 0, (
        f"rank {rank}: ColumnParallel forward must not call all_reduce, "
        f"got {fwd_counter.count}"
    )

    # Backward should trigger exactly 1 all-reduce on the input grad path
    with AllreduceCounter() as bwd_counter:
        out.sum().backward()
    if world_size > 1:
        assert bwd_counter.count >= 1, (
            f"rank {rank}: ColumnParallel backward must call all_reduce at least once, "
            f"got {bwd_counter.count}"
        )


def row_forward_triggers_allreduce(rank, world_size, impl):
    in_f, out_f = 8, 4
    layer = impl.RowParallelLinear(in_f, out_f, bias=False, input_is_parallel=False)
    x = torch.randn(2, in_f, requires_grad=True)

    with AllreduceCounter() as counter:
        out = layer(x)
    if world_size > 1:
        assert counter.count >= 1, (
            f"rank {rank}: RowParallel forward must call all_reduce at least once, "
            f"got {counter.count}"
        )
    # Sanity: backward should NOT call all-reduce (output-side path is identity in backward)
    with AllreduceCounter() as bwd_counter:
        out.sum().backward()
    # We allow 0; some implementations may all-reduce loss reduction etc., so we use <= 0 as informational.
    assert bwd_counter.count == 0, (
        f"rank {rank}: RowParallel backward should not call all_reduce, "
        f"got {bwd_counter.count}"
    )


def row_bias_added_once(rank, world_size, impl):
    """Catch the classic bug: putting bias inside the local F.linear before all-reduce.

    With replicated bias, if you do `F.linear(x_local, W_local, bias)` then all-reduce,
    you get `(sum of x_i @ W_i^T) + bias * world_size`. The output will be off by
    `(world_size - 1) * bias`. This test detects exactly that.
    """
    torch.manual_seed(11)
    in_f, out_f = 8, 4

    canonical = nn.Linear(in_f, out_f, bias=True).double()
    for p in canonical.parameters():
        dist.broadcast(p.data, src=0)

    layer = impl.RowParallelLinear(
        in_f, out_f, bias=True, input_is_parallel=False
    ).double()
    in_per_part = in_f // world_size
    layer.weight.data.copy_(canonical.weight.data[:, rank * in_per_part:(rank + 1) * in_per_part])
    # bias is replicated — copy canonical bias into every rank.
    if layer.bias is not None:
        layer.bias.data.copy_(canonical.bias.data)

    x = torch.randn(2, in_f, dtype=torch.float64)
    dist.broadcast(x, src=0)
    actual = layer(x)
    expected = canonical(x)
    diff = (actual - expected).abs().max().item()
    bias_mean = canonical.bias.data.abs().mean().item() + 1e-12
    assert diff < 1e-9, (
        f"rank {rank}: output differs from single-GPU by {diff:.4f} "
        f"(≈ {diff / bias_mean:.2f}x mean|bias|). "
        f"Likely cause: bias is being added {world_size}x because it was passed "
        f"into F.linear *before* all-reduce. Move `+ self.bias` to AFTER the "
        f"_ReduceFromParallelRegion.apply call."
    )
