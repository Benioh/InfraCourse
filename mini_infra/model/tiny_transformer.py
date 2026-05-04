from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TinyModelConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 128
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_torch_model(config: TinyModelConfig):
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the torch backend") from exc

    class TinyTransformerLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position = nn.Embedding(config.max_seq_len, config.hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                batch_first=True,
            )
            self.layers = nn.TransformerEncoder(layer, num_layers=config.num_layers)
            self.head = nn.Linear(config.hidden_size, config.vocab_size)

        def forward(self, input_ids):
            positions = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0)
            hidden = self.token(input_ids) + self.position(positions)
            hidden = self.layers(hidden)
            return self.head(hidden)

    return TinyTransformerLM()


def count_parameters_from_config(config: TinyModelConfig) -> int:
    embedding = config.vocab_size * config.hidden_size
    attention = config.num_layers * 4 * config.hidden_size * config.hidden_size
    mlp = config.num_layers * 8 * config.hidden_size * config.hidden_size
    output = config.hidden_size * config.vocab_size
    return embedding + attention + mlp + output
