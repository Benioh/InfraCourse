from __future__ import annotations

from typing import Iterable


def model_provider() -> dict[str, int | str]:
    return {"model_type": "GPT", "num_layers": 2, "hidden_size": 64, "vocab_size": 128}


def get_batch(data_iterator: Iterable[str]) -> dict[str, str]:
    text = next(data_iterator)
    return {"tokens": text, "labels": text}


def loss_func(output_tensor: dict[str, float]) -> float:
    return float(output_tensor["loss"])


def forward_step(
    data_iterator: Iterable[str], model: dict[str, int | str]
) -> dict[str, float | str]:
    batch = get_batch(data_iterator)
    token_count = len(batch["tokens"].split())
    loss = round(1.0 / max(token_count, 1), 6)
    return {
        "loss": loss,
        "model_type": str(model["model_type"]),
        "token_count": token_count,
    }
