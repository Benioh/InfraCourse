"""L05.3 Patch tests · CPU first, GPU optional."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    return importlib.import_module(f"{os.environ.get('IMPL', 'starter')}.fsdp2_wrap")


class _TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.nn.functional.relu(self.attn(x)))


class _TinyLlama(nn.Module):
    def __init__(self, dim: int = 32, depth: int = 4):
        super().__init__()
        self.embed = nn.Embedding(64, dim)
        self.blocks = nn.ModuleList([_TinyBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, 64, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class _MockMpPolicy:
    def __init__(self, param_dtype, reduce_dtype, output_dtype=None):
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.output_dtype = output_dtype


def _spy_fully_shard(record: list[tuple[str, dict]]):
    def fn(module, **kwargs):
        record.append((module.__class__.__name__, kwargs))
        return module

    return fn


def test_wrap_marks_each_block():
    impl = _impl()
    model = _TinyLlama(depth=3)
    record: list[tuple[str, dict]] = []
    report = impl.wrap_transformer_blocks_fsdp2(
        model, _TinyBlock, _fully_shard=_spy_fully_shard(record)
    )
    assert report.wrapped_blocks == ["blocks.0", "blocks.1", "blocks.2"]
    assert sum(1 for cls, _ in record if cls == "_TinyBlock") == 3


def test_wrap_marks_root_last():
    impl = _impl()
    model = _TinyLlama(depth=2)
    record: list[tuple[str, dict]] = []
    impl.wrap_transformer_blocks_fsdp2(
        model, _TinyBlock, _fully_shard=_spy_fully_shard(record)
    )
    assert record[-1][0] == "_TinyLlama"
    assert all(cls == "_TinyBlock" for cls, _ in record[:-1])


def test_mp_policy_propagates():
    impl = _impl()
    model = _TinyLlama(depth=1)
    record: list[tuple[str, dict]] = []
    policy = _MockMpPolicy(param_dtype="bf16", reduce_dtype="fp32")
    report = impl.wrap_transformer_blocks_fsdp2(
        model, _TinyBlock, mp_policy=policy, _fully_shard=_spy_fully_shard(record)
    )
    for _cls, kwargs in record:
        assert kwargs.get("mp_policy") is policy
    assert report.mp_policy_summary["param_dtype"] == "bf16"


def test_skip_blocks_filter():
    impl = _impl()
    model = _TinyLlama(depth=4)
    record: list[tuple[str, dict]] = []
    skipped = ["blocks.1", "blocks.3"]
    report = impl.wrap_transformer_blocks_fsdp2(
        model,
        _TinyBlock,
        skip=lambda name, _module: name in skipped,
        _fully_shard=_spy_fully_shard(record),
    )
    assert report.wrapped_blocks == ["blocks.0", "blocks.2"]


def test_reshard_after_forward_default_true():
    impl = _impl()
    model = _TinyLlama(depth=1)
    record: list[tuple[str, dict]] = []
    impl.wrap_transformer_blocks_fsdp2(model, _TinyBlock, _fully_shard=_spy_fully_shard(record))
    for _cls, kwargs in record:
        assert kwargs.get("reshard_after_forward") is True


# ---- GPU smoke tests (require CUDA + torch >= 2.4 + a process group) ----

def _maybe_init_pg() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29555")
            dist.init_process_group("nccl", rank=0, world_size=1)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.gpu
def test_forward_backward_smoke():
    impl = _impl()
    if not _maybe_init_pg():
        pytest.skip("requires CUDA + a 1-process NCCL group")
    try:
        from torch.distributed._composable.fsdp import (
            MixedPrecisionPolicy,
            fully_shard,
        )
    except ImportError:
        pytest.skip("torch >= 2.4 required for FSDP2 fully_shard")

    device = torch.device("cuda:0")
    model = _TinyLlama(dim=64, depth=3).to(device)
    impl.wrap_transformer_blocks_fsdp2(
        model,
        _TinyBlock,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        _fully_shard=fully_shard,
    )
    ids = torch.randint(0, 64, (2, 16), device=device)
    logits = model(ids)
    loss = logits.float().mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.gpu
def test_state_dict_round_trip(tmp_path: Path):
    impl = _impl()
    if not _maybe_init_pg():
        pytest.skip("requires CUDA + a 1-process NCCL group")
    try:
        from torch.distributed._composable.fsdp import fully_shard
    except ImportError:
        pytest.skip("torch >= 2.4 required for FSDP2 fully_shard")
    device = torch.device("cuda:0")
    model = _TinyLlama(dim=32, depth=2).to(device)
    impl.wrap_transformer_blocks_fsdp2(model, _TinyBlock, _fully_shard=fully_shard)
    state = {key: value.detach().clone().cpu() for key, value in model.state_dict().items()}
    sd_path = tmp_path / "ckpt.pt"
    torch.save(state, sd_path)
    loaded = torch.load(sd_path, map_location="cpu")
    assert set(loaded.keys()) == set(state.keys())
    for key in state:
        assert torch.allclose(state[key], loaded[key])
