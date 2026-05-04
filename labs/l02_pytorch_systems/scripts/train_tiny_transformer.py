from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import (
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_text,
    write_yaml,
)

MISSION_ID = "l02_pytorch_systems"
LAB_DIR = Path(__file__).resolve().parents[1]


class SyntheticTokens(Dataset):
    def __init__(
        self, seq_len: int, vocab_size: int, sleep_ms: int = 0, size: int = 4096
    ) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.sleep_ms = sleep_ms
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sleep_ms:
            time.sleep(self.sleep_ms / 1000)
        generator = torch.Generator().manual_seed(index)
        x = torch.randint(0, self.vocab_size, (self.seq_len,), generator=generator)
        y = torch.roll(x, shifts=-1)
        return x, y


class TinyDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        checkpoint_layers: bool,
    ) -> None:
        super().__init__()
        self.checkpoint_layers = checkpoint_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(2048, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        hidden = self.embed(tokens) + self.pos(positions)
        causal_mask = torch.triu(
            torch.ones(
                tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        for layer in self.layers:
            if self.checkpoint_layers and self.training:
                hidden = checkpoint(layer, hidden, causal_mask, use_reentrant=False)
            else:
                hidden = layer(hidden, src_mask=causal_mask)
        return self.head(self.norm(hidden))


def load_config(path: str) -> dict:
    return yaml.safe_load((LAB_DIR / path).read_text(encoding="utf-8"))


def make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_precision(config: dict) -> tuple[torch.dtype | None, bool]:
    precision = config.get("precision", "fp32")
    if precision == "bf16" and torch.cuda.is_available():
        return torch.bfloat16, True
    if precision == "fp16" and torch.cuda.is_available():
        return torch.float16, True
    return None, False


def maybe_profile(run_dir: Path, model: nn.Module, batch: torch.Tensor) -> None:
    profiler_dir = run_dir / "artifacts" / "profiler"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        with record_function("warmup_forward"):
            model(batch)
    prof.export_chrome_trace(str(profiler_dir / "trace.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = make_device()
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(run_dir / "config.resolved.yaml", config)

    dataset = SyntheticTokens(
        config["seq_len"], config["vocab_size"], config.get("dataloader_sleep_ms", 0)
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    iterator = iter(loader)

    model = TinyDecoder(
        config["vocab_size"],
        config["d_model"],
        config["n_heads"],
        config["n_layers"],
        config.get("activation_checkpointing", False),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    autocast_dtype, use_autocast = set_precision(config)
    grad_norm_value = 0.0

    warmup_batch, _ = next(iterator)
    maybe_profile(run_dir, model, warmup_batch.to(device))
    iterator = iter(loader)

    log_lines = [f"[{utc_now()}] device={device.type}"]
    for step in range(1, config["steps"] + 1):
        step_start = time.perf_counter()

        dl_start = time.perf_counter()
        try:
            tokens, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            tokens, targets = next(iterator)
        dataloader_ms = (time.perf_counter() - dl_start) * 1000

        tokens = tokens.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()
        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits = model(tokens)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
        else:
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        forward_ms = (time.perf_counter() - forward_start) * 1000

        backward_start = time.perf_counter()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_value = float(total_norm)
        backward_ms = (time.perf_counter() - backward_start) * 1000

        optimizer_start = time.perf_counter()
        optimizer.step()
        optimizer_ms = (time.perf_counter() - optimizer_start) * 1000
        step_ms = (time.perf_counter() - step_start) * 1000

        tokens_per_sec = (tokens.numel() / step_ms) * 1000
        peak_memory_gb = (
            round(torch.cuda.max_memory_allocated(device) / 1024**3, 4)
            if torch.cuda.is_available()
            else 0.0
        )

        row = {
            "step": step,
            "timestamp": utc_now(),
            "metric_type": "train",
            "loss": round(float(loss.item()), 6),
            "step_time_ms": round(step_ms, 3),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "peak_memory_gb": peak_memory_gb,
            "dataloader_time_ms": round(dataloader_ms, 3),
            "forward_time_ms": round(forward_ms, 3),
            "backward_time_ms": round(backward_ms, 3),
            "optimizer_time_ms": round(optimizer_ms, 3),
            "grad_norm": round(grad_norm_value, 4),
        }
        append_jsonl(run_dir / "metrics.jsonl", row)
        log_lines.append(json.dumps(row))

    write_text(run_dir / "train.log", "\n".join(log_lines) + "\n")
    last_loss = json.loads(log_lines[-1])["loss"]
    write_text(
        run_dir / "report.md",
        f"# Mission Report：{MISSION_ID}\n\n"
        "## 1. 目标\n\n"
        "用显式 timing 观测 tiny transformer 的显存和吞吐行为。\n\n"
        "## 2. 环境与配置\n"
        f"- GPU: {device.type}\n"
        "- 框架：PyTorch\n"
        "- Model: tiny decoder-only transformer\n"
        "- 数据集：合成 token\n"
        f"- Precision: {config['precision']}\n"
        f"- 并行：单设备，activation checkpointing={config['activation_checkpointing']}\n\n"
        "## 3. 预测\n"
        "- 预测瓶颈：激活显存\n"
        "- 预测显存：中等，且对序列长度敏感\n"
        "- 预测吞吐：warmup 后稳定\n"
        "- 预测失败：长序列 OOM 或 dataloader 阻塞\n\n"
        "## 4. 运行命令\n\n"
        "见 `command.sh`。\n\n"
        "## 5. 结果\n"
        f"- Loss： {last_loss}\n"
        "- 吞吐：见 `metrics.jsonl`\n"
        "- 显存：见 `peak_memory_gb`\n"
        "- 其它指标：已记录 dataloader、forward、backward、optimizer timing\n\n"
        "## 6. 诊断\n\n"
        "本实验把 dataloader、forward、backward、optimizer 拆开计时，让瓶颈定位有证据而不是靠猜。\n\n"
        "## 7. Debug 工单\n"
        "- Ticket：pt_oom_001\n"
        "- 根因：激活显存随序列长度增长\n"
        "- 最小修复： lower sequence length or enable checkpointing\n"
        "- 验证方式： compare `peak_memory_gb` and step stability across configs\n\n"
        "## 8. PR Review\n"
        "- 审查的 patch： silent batch-size or dtype changes\n"
        "- 风险： invalid throughput comparison\n"
        "- 增加的测试： metrics emission plus profiler trace existence\n\n"
        "## 9. 我原来误解了什么\n\n"
        "## 10. 如果迁移到 8×H200\n\n"
        "在更长序列和分布式训练中保持同样的观测纪律。\n\n"
        "## 11. 下一步\n\n"
        "进入分布式原语或 Megatron 数据准备。\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
