from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Any, Callable

from mini_infra.data.toy_data import read_jsonl, sample_text
from mini_infra.megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from mini_infra.megatron.core.pipeline_parallel.schedules import (
    bubble_ratio,
    get_forward_backward_func,
)
from mini_infra.observability.io import (
    append_jsonl,
    mini_run_dir,
    write_json,
    write_yaml,
)

from .arguments import MiniMegatronArgs
from .checkpointing import save_checkpoint
from .initialize import initialize_megatron


class TrainStepError(ValueError):
    """Raised when the forward/backward result does not match the train-step contract."""


class MiniLRScheduler:
    """Small scheduler object with Megatron-like step/state boundaries."""

    def __init__(self, optimizer: Any, decay: float = 0.98) -> None:
        self.optimizer = optimizer
        self.decay = decay
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        for group in self.optimizer.param_groups:
            group["lr"] *= self.decay

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict[str, float | int]:
        return {"step_count": self.step_count, "lr": self.get_lr()}


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
        grad_norm = result.get("grad_norm")
        return bool(result.get("success", True)), (
            float(grad_norm) if grad_norm is not None else None
        )
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
    data_iterator,
    model: dict,
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
        metrics["grad_norm"] = grad_norm
    for key, value in output.items():
        if key not in {"loss", "losses"}:
            metrics[key] = value
    return metrics


def training_log(run_dir: Path, metrics: dict) -> None:
    append_jsonl(
        run_dir / "metrics.jsonl", {"metric_type": "mini_megatron_train", **metrics}
    )


def pretrain(
    args: MiniMegatronArgs,
    model_provider: Callable,
    forward_step_func: Callable,
    run_id: str | None = None,
) -> dict:
    run_dir = mini_run_dir("megatron", run_id)
    state = initialize_megatron(args)
    rows = read_jsonl(Path(args.data_path))
    texts = [sample_text(row) for row in rows]
    data_iterator = cycle(texts)
    model = model_provider()
    optimizer = DistributedOptimizer(
        parameter_count=114688, data_parallel_size=state.data_parallel_size
    )
    lr_scheduler = MiniLRScheduler(optimizer)
    schedule = get_forward_backward_func(args.pipeline_model_parallel_size)
    schedule_events = schedule(
        num_microbatches=max(args.global_batch_size // max(args.micro_batch_size, 1), 1)
    )
    final_metrics = {}
    for iteration in range(1, args.train_samples + 1):
        final_metrics = train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            lr_scheduler,
            iteration,
        )
        final_metrics.update(
            {
                "tp": args.tensor_model_parallel_size,
                "pp": args.pipeline_model_parallel_size,
                "sequence_parallel": args.sequence_parallel,
                "distributed_optimizer": args.use_distributed_optimizer,
                "pipeline_bubble_ratio": bubble_ratio(
                    args.global_batch_size // max(args.micro_batch_size, 1),
                    args.pipeline_model_parallel_size,
                ),
                "optimizer_state_memory_ratio": optimizer.state_memory_ratio(),
            }
        )
        training_log(run_dir, final_metrics)
    checkpoint = save_checkpoint(
        Path(args.save),
        int(final_metrics["iteration"]),
        model_state={"provider": "model_provider", **model},
        optimizer_state=optimizer.state_dict(),
        scheduler_state=lr_scheduler.state_dict(),
        parallel_state=state.__dict__,
    )
    write_yaml(run_dir / "config.resolved.yaml", args.to_dict())
    write_json(
        run_dir / "artifacts" / "schedule.json",
        [event.__dict__ for event in schedule_events],
    )
    write_json(run_dir / "artifacts" / "checkpoint.json", checkpoint)
    return {
        "run_dir": str(run_dir),
        "final_metrics": final_metrics,
        "checkpoint": checkpoint,
    }
