"""L04.8 Patch · Megatron-shaped train_step.

This is a small, testable slice of ``mini_infra/megatron/training/training.py``:
zero grads, run forward/backward, step optimizer, step LR scheduler, return metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


class TrainStepError(ValueError):
    """Raised when the forward/backward result does not match the train-step contract."""


def train_step(
    forward_backward_func: Callable[[Any, Any], dict[str, Any]],
    data_iterator: Iterable[Any],
    model: Any,
    optimizer: Any,
    lr_scheduler: Any | None,
    iteration: int,
) -> dict[str, Any]:
    """Run one Megatron-shaped training step.

    ``forward_backward_func`` returns either ``{"loss": float}`` or
    ``{"losses": [float, ...]}``. ``optimizer.step()`` may return:
    - bool: True means update succeeded, False means skipped/overflow.
    - dict: ``{"success": bool, "grad_norm": float}``.
    """
    # TODO(student): call optimizer.zero_grad() before forward/backward when present.
    # TODO(student): run forward_backward_func(data_iterator, model).
    # TODO(student): average microbatch losses and validate at least one loss exists.
    # TODO(student): call optimizer.step(), detect skipped iterations, and expose grad_norm.
    # TODO(student): call lr_scheduler.step() only when optimizer update succeeds.
    # TODO(student): return iteration/loss/skipped_iter/lr/grad_norm metrics.
    raise NotImplementedError("L04.8: implement train_step")
