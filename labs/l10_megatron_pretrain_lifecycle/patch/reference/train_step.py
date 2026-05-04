"""Reference solution for L04.8 Patch."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


class TrainStepError(ValueError):
    """Raised when the forward/backward result does not match the train-step contract."""


def _as_loss_list(output: dict[str, Any]) -> list[float]:
    if "losses" in output:
        losses = list(output["losses"])
    elif "loss" in output:
        losses = [output["loss"]]
    else:
        raise TrainStepError("forward_backward_func must return loss or losses")
    if not losses:
        raise TrainStepError("at least one microbatch loss is required")
    return [float(loss) for loss in losses]


def _parse_step_result(result: Any) -> tuple[bool, float | None]:
    if isinstance(result, dict):
        return bool(result.get("success", True)), result.get("grad_norm")
    if isinstance(result, bool):
        return result, None
    if result is None:
        return True, None
    raise TrainStepError("optimizer.step() must return bool, dict, or None")


def _current_lr(optimizer: Any, lr_scheduler: Any | None) -> float | None:
    if lr_scheduler is not None and hasattr(lr_scheduler, "get_lr"):
        return float(lr_scheduler.get_lr())
    groups = getattr(optimizer, "param_groups", None)
    if groups:
        return float(groups[0].get("lr", 0.0))
    return None


def train_step(
    forward_backward_func: Callable[[Any, Any], dict[str, Any]],
    data_iterator: Iterable[Any],
    model: Any,
    optimizer: Any,
    lr_scheduler: Any | None,
    iteration: int,
) -> dict[str, Any]:
    if hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()

    output = forward_backward_func(data_iterator, model)
    if not isinstance(output, dict):
        raise TrainStepError("forward_backward_func must return a dict")
    losses = _as_loss_list(output)
    loss = sum(losses) / len(losses)

    success, grad_norm = _parse_step_result(optimizer.step())
    if success and lr_scheduler is not None:
        lr_scheduler.step()

    metrics: dict[str, Any] = {
        "iteration": int(iteration),
        "loss": loss,
        "num_microbatches": len(losses),
        "skipped_iter": 0 if success else 1,
        "lr": _current_lr(optimizer, lr_scheduler),
    }
    if grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm)
    if "tokens" in output:
        metrics["tokens"] = int(output["tokens"])
    return metrics
