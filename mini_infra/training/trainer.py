from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mini_infra.model.tiny_transformer import (
    TinyModelConfig,
    count_parameters_from_config,
)
from mini_infra.observability.io import append_jsonl, utc_now, write_json


@dataclass
class TrainConfig:
    backend: str = "simulated"
    steps: int = 12
    batch_size: int = 4
    seq_len: int = 64
    learning_rate: float = 3e-4
    seed: int = 2026

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MiniTrainer:
    def __init__(
        self, model_config: TinyModelConfig, train_config: TrainConfig, run_dir: Path
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"

    def run(self) -> dict[str, Any]:
        if self.train_config.backend == "torch":
            return self._run_torch()
        return self._run_simulated()

    def _run_simulated(self) -> dict[str, Any]:
        parameter_count = count_parameters_from_config(self.model_config)
        losses = []
        for step in range(1, self.train_config.steps + 1):
            loss = round(3.2 / (1 + step * 0.08), 4)
            tokens_per_sec = round(
                9000
                + step * 35
                + self.train_config.batch_size * self.train_config.seq_len,
                2,
            )
            losses.append(loss)
            append_jsonl(
                self.metrics_path,
                {
                    "timestamp": utc_now(),
                    "step": step,
                    "metric_type": "train",
                    "loss": loss,
                    "tokens_per_sec": tokens_per_sec,
                    "backend": "simulated",
                },
            )
        checkpoint_dir = self.run_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            checkpoint_dir / "weights.json",
            {
                "note": "simulated weights for MiniInfra smoke",
                "parameter_count_estimate": parameter_count,
                "final_loss": losses[-1],
            },
        )
        return {
            "backend": "simulated",
            "final_loss": losses[-1],
            "parameter_count": parameter_count,
            "checkpoint_dir": str(checkpoint_dir),
        }

    def _run_torch(self) -> dict[str, Any]:
        import torch
        from torch import nn

        from mini_infra.model.tiny_transformer import build_torch_model

        torch.manual_seed(self.train_config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_torch_model(self.model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.train_config.learning_rate
        )
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        for step in range(1, self.train_config.steps + 1):
            input_ids = torch.randint(
                0,
                self.model_config.vocab_size,
                (self.train_config.batch_size, self.train_config.seq_len),
                device=device,
            )
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            append_jsonl(
                self.metrics_path,
                {
                    "timestamp": utc_now(),
                    "step": step,
                    "metric_type": "train",
                    "loss": round(loss_value, 6),
                    "tokens_per_sec": None,
                    "backend": "torch",
                    "device": str(device),
                },
            )
        checkpoint_dir = self.run_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": self.model_config.to_dict(),
                "train_config": self.train_config.to_dict(),
            },
            checkpoint_dir / "checkpoint.pt",
        )
        return {
            "backend": "torch",
            "final_loss": round(losses[-1], 6),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "checkpoint_dir": str(checkpoint_dir),
        }
