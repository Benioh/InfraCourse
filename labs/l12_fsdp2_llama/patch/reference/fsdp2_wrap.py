"""Reference solution for L13 Patch."""

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


def _resolve_fully_shard(provided: Callable[..., Any] | None) -> Callable[..., Any]:
    if provided is not None:
        return provided
    from torch.distributed._composable.fsdp import fully_shard  # type: ignore[import-not-found]

    return fully_shard


def wrap_transformer_blocks_fsdp2(
    model: nn.Module,
    block_cls: type[nn.Module],
    mp_policy: Any | None = None,
    reshard_after_forward: bool = True,
    skip: Callable[[str, nn.Module], bool] | None = None,
    *,
    _fully_shard: Callable[..., Any] | None = None,
) -> WrapReport:
    fully_shard = _resolve_fully_shard(_fully_shard)
    wrapped_blocks: list[str] = []
    for name, child in model.named_modules():
        if name == "":
            continue
        if not isinstance(child, block_cls):
            continue
        if skip is not None and skip(name, child):
            continue
        kwargs: dict[str, Any] = {"reshard_after_forward": reshard_after_forward}
        if mp_policy is not None:
            kwargs["mp_policy"] = mp_policy
        fully_shard(child, **kwargs)
        wrapped_blocks.append(name)
    root_kwargs: dict[str, Any] = {"reshard_after_forward": reshard_after_forward}
    if mp_policy is not None:
        root_kwargs["mp_policy"] = mp_policy
    fully_shard(model, **root_kwargs)
    return WrapReport(
        wrapped_blocks=wrapped_blocks,
        root_wrapped=True,
        mp_policy_summary=_summarize_policy(mp_policy),
        reshard_after_forward=reshard_after_forward,
    )
