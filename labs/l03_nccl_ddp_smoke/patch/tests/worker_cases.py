"""Per-rank assertions for L01.5 ManualDDP tests."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def make_mlp(in_dim=8, hidden=16, out_dim=4):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


def grads_match_pytorch_ddp(rank, world_size, impl):
    """Build same model on all ranks, give each rank a different x; the grads after
    synchronize_grads() must equal what torch.nn.parallel.DDP would produce."""
    from torch.nn.parallel import DistributedDataParallel

    torch.manual_seed(42)
    canonical = make_mlp()
    for p in canonical.parameters():
        dist.broadcast(p.data, src=0)

    # Each rank uses a different input (this is what makes DDP non-trivial)
    x = torch.randn(2, 8) + rank * 0.1

    # ---- Reference: PyTorch DDP ----
    ref_model = make_mlp()
    for ref_p, can_p in zip(ref_model.parameters(), canonical.parameters()):
        ref_p.data.copy_(can_p.data)
    ref_ddp = DistributedDataParallel(ref_model)
    ref_ddp(x).sum().backward()
    ref_grads = [p.grad.clone() for p in ref_model.parameters()]

    # ---- Student impl ----
    stu_model = make_mlp()
    for stu_p, can_p in zip(stu_model.parameters(), canonical.parameters()):
        stu_p.data.copy_(can_p.data)
    stu_ddp = impl.ManualDDP(stu_model)
    stu_ddp.module(x).sum().backward()
    stu_ddp.synchronize_grads()

    for i, (ref_g, p) in enumerate(zip(ref_grads, stu_model.parameters())):
        diff = (p.grad - ref_g).abs().max().item()
        assert diff < 1e-6, f"rank {rank} param {i}: grad diff = {diff}"


def grads_are_averaged_not_summed(rank, world_size, impl):
    """If you forget the /world_size, grads are world_size× too large.
    This test catches that by comparing magnitude to single-GPU baseline."""
    torch.manual_seed(7)

    # Build an identical model on all ranks
    model = make_mlp()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # All ranks use SAME input — so each local grad equals what single-GPU would produce
    x = torch.randn(2, 8)
    dist.broadcast(x, src=0)

    # Compute the single-GPU reference grad on a clone (no DDP involved)
    ref_model = make_mlp()
    for ref_p, p in zip(ref_model.parameters(), model.parameters()):
        ref_p.data.copy_(p.data)
    ref_model(x).sum().backward()
    ref_grads = [p.grad.clone() for p in ref_model.parameters()]

    # Now run ManualDDP on student model
    ddp = impl.ManualDDP(model)
    ddp.module(x).sum().backward()
    ddp.synchronize_grads()

    # Average of identical local grads is just that grad — matches single-GPU.
    # If student forgot to divide by world_size, grad would be world_size× too big.
    for i, (ref_g, p) in enumerate(zip(ref_grads, model.parameters())):
        diff = (p.grad - ref_g).abs().max().item()
        assert diff < 1e-6, (
            f"rank {rank} param {i}: grad differs from single-GPU by {diff}. "
            f"Did you forget to divide by world_size?"
        )


def world_size_1_is_noop(rank, world_size, impl):
    """When run with only one process, synchronize_grads should not change anything."""
    if world_size != 1:
        return  # only meaningful in 1-rank run
    torch.manual_seed(11)
    model = make_mlp()
    x = torch.randn(2, 8)
    model(x).sum().backward()
    before = [p.grad.clone() for p in model.parameters()]
    ddp = impl.ManualDDP(model)
    ddp.synchronize_grads()
    after = [p.grad.clone() for p in model.parameters()]
    for b, a in zip(before, after):
        assert torch.equal(a, b)


def skips_no_grad_params(rank, world_size, impl):
    """Params with requires_grad=False must not be touched."""
    torch.manual_seed(13)
    model = make_mlp()
    # Freeze the second linear
    for p in model[2].parameters():
        p.requires_grad_(False)

    x = torch.randn(2, 8)
    model(x).sum().backward()

    ddp = impl.ManualDDP(model)
    ddp.synchronize_grads()

    # Frozen params should have grad == None
    for p in model[2].parameters():
        assert p.grad is None, "frozen param should have no grad"


def handles_partial_grads(rank, world_size, impl):
    """If any param has grad=None (e.g. a branch wasn't used), don't crash."""
    torch.manual_seed(17)
    model = make_mlp()
    x = torch.randn(2, 8)
    model(x).sum().backward()

    # Manually set one param's grad to None to simulate dead path
    list(model.parameters())[0].grad = None

    ddp = impl.ManualDDP(model)
    # Must not raise
    ddp.synchronize_grads()
