"""Per-rank assertions for L05 BucketedManualDDP."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def _make_model(small=False):
    if small:
        return nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    )


def grads_match_pytorch_ddp(rank, world_size, impl):
    from torch.nn.parallel import DistributedDataParallel

    torch.manual_seed(42)
    canonical = _make_model()
    for p in canonical.parameters():
        dist.broadcast(p.data, src=0)

    x = torch.randn(2, 64) + rank * 0.1

    # PyTorch DDP reference
    ref = _make_model()
    for p_ref, p_can in zip(ref.parameters(), canonical.parameters()):
        p_ref.data.copy_(p_can.data)
    ref_ddp = DistributedDataParallel(ref)
    ref_ddp(x).sum().backward()
    ref_grads = [p.grad.clone() for p in ref.parameters()]

    # Bucketed impl
    stu = _make_model()
    for p_stu, p_can in zip(stu.parameters(), canonical.parameters()):
        p_stu.data.copy_(p_can.data)
    bddp = impl.BucketedManualDDP(stu, bucket_size_mb=0.05)  # small bucket → multiple buckets
    bddp.module(x).sum().backward()
    bddp.synchronize_grads()

    for i, (g_ref, p) in enumerate(zip(ref_grads, stu.parameters())):
        diff = (p.grad - g_ref).abs().max().item()
        assert diff < 1e-6, f"rank {rank} param {i}: diff = {diff}"


def works_with_huge_param(rank, world_size, impl):
    """A single Linear with big out_features so weight > bucket size; must still work."""
    torch.manual_seed(7)
    model = nn.Linear(32, 4096)  # weight = 32*4096*4 = 512KB
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    x = torch.randn(2, 32) + rank * 0.05

    bddp = impl.BucketedManualDDP(model, bucket_size_mb=0.001)  # 1KB bucket; weight is ~500x larger
    bddp.module(x).sum().backward()
    bddp.synchronize_grads()

    # Should not crash; sanity-check grad finite
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def works_with_tiny_params(rank, world_size, impl):
    """Many small layers; final grads still match a non-bucketed reference."""
    torch.manual_seed(11)

    def _build():
        # 10 tiny linears
        return nn.Sequential(*[nn.Linear(8, 8) for _ in range(10)])

    canonical = _build()
    for p in canonical.parameters():
        dist.broadcast(p.data, src=0)

    x = torch.randn(2, 8) + rank * 0.07

    # Single-rank reference: same input on all ranks, just compute single-GPU grad
    ref = _build()
    for r, c in zip(ref.parameters(), canonical.parameters()):
        r.data.copy_(c.data)
    # Use SAME x on all ranks so reference equals what synchronize_grads should produce
    same_x = torch.randn(2, 8)
    dist.broadcast(same_x, src=0)
    ref(same_x).sum().backward()
    ref_grads = [p.grad.clone() for p in ref.parameters()]

    stu = _build()
    for s, c in zip(stu.parameters(), canonical.parameters()):
        s.data.copy_(c.data)
    bddp = impl.BucketedManualDDP(stu, bucket_size_mb=10)  # All fit in one bucket
    bddp.module(same_x).sum().backward()
    bddp.synchronize_grads()
    for i, (g_ref, p) in enumerate(zip(ref_grads, stu.parameters())):
        diff = (p.grad - g_ref).abs().max().item()
        assert diff < 1e-6, f"rank {rank} param {i}: diff={diff}"


def world_size_1_is_noop(rank, world_size, impl):
    if world_size != 1:
        return
    torch.manual_seed(13)
    model = _make_model(small=True)
    x = torch.randn(2, 8)
    model(x).sum().backward()
    before = [p.grad.clone() for p in model.parameters()]
    bddp = impl.BucketedManualDDP(model)
    bddp.synchronize_grads()
    for b, p in zip(before, model.parameters()):
        assert torch.equal(b, p.grad)


def skips_no_grad_params(rank, world_size, impl):
    torch.manual_seed(17)
    model = _make_model(small=True)
    # Freeze last linear
    for p in model[2].parameters():
        p.requires_grad_(False)

    x = torch.randn(2, 8)
    model(x).sum().backward()

    bddp = impl.BucketedManualDDP(model)
    bddp.synchronize_grads()

    for p in model[2].parameters():
        assert p.grad is None
