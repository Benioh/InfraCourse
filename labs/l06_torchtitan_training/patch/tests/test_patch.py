"""L03 Patch tests · CPU OK; one GPU-only memory test auto-skips."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PATCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATCH_DIR))


def _impl():
    name = os.environ.get("IMPL", "starter")
    return importlib.import_module(f"{name}.selective_ckpt")


class _AttentionLike(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(x).relu()


class _MLPLike(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())


class _Net(nn.Module):
    """Has named children: attn0, mlp0, attn1, mlp1."""

    def __init__(self, dim=16):
        super().__init__()
        self.attn0 = _AttentionLike(dim)
        self.mlp0 = _MLPLike(dim)
        self.attn1 = _AttentionLike(dim)
        self.mlp1 = _MLPLike(dim)

    def forward(self, x):
        x = self.attn0(x)
        x = self.mlp0(x)
        x = self.attn1(x)
        x = self.mlp1(x)
        return x


def _clone_model(model: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(model)


def test_output_matches_no_ckpt():
    impl = _impl()
    torch.manual_seed(42)
    base = _Net()
    wrapped = _clone_model(base)
    impl.selective_checkpoint_wrap(wrapped, impl.attention_only_policy)

    x = torch.randn(2, 4, 16, requires_grad=True)
    expected = base(x).sum()
    actual = wrapped(x.clone().detach().requires_grad_(True)).sum()
    assert torch.allclose(actual, expected, atol=1e-5)


def test_grads_match_no_ckpt():
    impl = _impl()
    torch.manual_seed(42)
    base = _Net()
    wrapped = _clone_model(base)
    impl.selective_checkpoint_wrap(wrapped, impl.attention_only_policy)

    x = torch.randn(2, 4, 16)

    base_x = x.clone().detach().requires_grad_(True)
    base(base_x).sum().backward()

    wrap_x = x.clone().detach().requires_grad_(True)
    wrapped(wrap_x).sum().backward()

    assert torch.allclose(base_x.grad, wrap_x.grad, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(base.named_parameters(), wrapped.named_parameters()):
        # Names may differ (wrapped has "attn0.module.proj.weight"); compare values
        if p1.grad is not None and p2.grad is not None:
            assert torch.allclose(p1.grad, p2.grad, atol=1e-5), (
                f"grad mismatch on {n1} vs {n2}"
            )


def test_attention_only_policy_counts():
    impl = _impl()
    model = _Net()
    impl.selective_checkpoint_wrap(model, impl.attention_only_policy)
    # attn0 and attn1 should be wrapped, mlp0 and mlp1 not
    from importlib import import_module
    base_mod = import_module(f"{os.environ.get('IMPL', 'starter')}.selective_ckpt")
    Wrapper = base_mod._CheckpointWrapper

    assert isinstance(model.attn0, Wrapper)
    assert isinstance(model.attn1, Wrapper)
    assert not isinstance(model.mlp0, Wrapper)
    assert not isinstance(model.mlp1, Wrapper)


def test_all_false_policy_is_noop():
    impl = _impl()
    model = _Net()
    types_before = {n: type(c) for n, c in model.named_children()}
    impl.selective_checkpoint_wrap(model, lambda name, m: False)
    types_after = {n: type(c) for n, c in model.named_children()}
    assert types_before == types_after


@pytest.mark.gpu
def test_memory_drops_with_attn_ckpt():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    impl = _impl()
    torch.manual_seed(7)

    base = _Net(dim=512).cuda()
    wrapped = _clone_model(base)
    impl.selective_checkpoint_wrap(wrapped, impl.attention_only_policy)

    x = torch.randn(8, 32, 512, device="cuda", requires_grad=True)

    torch.cuda.reset_peak_memory_stats()
    base(x).sum().backward()
    base_peak = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    wrapped(x.detach().clone().requires_grad_(True)).sum().backward()
    wrap_peak = torch.cuda.max_memory_allocated()

    # Wrapped should use less peak memory (we recompute attention activations)
    assert wrap_peak < base_peak, f"expected wrap_peak < base_peak, got {wrap_peak} vs {base_peak}"
