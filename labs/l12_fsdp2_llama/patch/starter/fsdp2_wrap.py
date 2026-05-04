"""L05.3 Patch · Wrap a Llama-style model with FSDP2 ``fully_shard``.

Implementation must NOT use ``torch.distributed.fsdp.FullyShardedDataParallel``
(that is FSDP1, with flat-parameter semantics). Use the composable FSDP2 API:

    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

For testability we also let the caller inject a fake ``fully_shard`` impl via
the ``_fully_shard`` argument; the production path imports it from torch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass
class WrapReport:
    wrapped_blocks: list[str]
    root_wrapped: bool
    mp_policy_summary: dict[str, Any]
    reshard_after_forward: bool


def _summarize_policy(mp_policy: Any) -> dict[str, Any]:
    if mp_policy is None:
        return {"present": False}
    return {
        "present": True,
        "param_dtype": str(getattr(mp_policy, "param_dtype", None)),
        "reduce_dtype": str(getattr(mp_policy, "reduce_dtype", None)),
        "output_dtype": str(getattr(mp_policy, "output_dtype", None)),
    }


def wrap_transformer_blocks_fsdp2(
    model: nn.Module,
    block_cls: type[nn.Module],
    mp_policy: Any | None = None,
    reshard_after_forward: bool = True,
    skip: Callable[[str, nn.Module], bool] | None = None,
    *,
    _fully_shard: Callable[..., Any] | None = None,
) -> WrapReport:
    """Wrap each ``block_cls`` instance and then the root ``model`` with FSDP2."""
    # TODO(student): if _fully_shard is None, import it from torch.distributed._composable.fsdp
    # TODO(student): walk model.named_modules() and collect every (name, child) where
    #               isinstance(child, block_cls) and not skip(name, child)
    # TODO(student): for each collected (name, child), call _fully_shard(child, mp_policy=..., reshard_after_forward=...)
    # TODO(student): finally call _fully_shard(model, ...) once on root
    # TODO(student): return a WrapReport listing the wrapped blocks (in traversal order),
    #               root_wrapped=True, the mp policy summary, and reshard_after_forward
    raise NotImplementedError("L05.3: implement wrap_transformer_blocks_fsdp2")
