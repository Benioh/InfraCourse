"""L04.8 · Drive a Megatron-shaped pretrain lifecycle using the student's patch.

This script demonstrates that the patched ``train_step`` actually drives a real
optimization loop: loss decreases, LR schedule fires, checkpoints are written,
metrics.jsonl is produced. It uses a tiny Llama-style transformer on synthetic
tokens so it runs on CPU; with ``--config configs/h200_125m.yaml`` it prints the
expected ``torchrun pretrain_gpt.py`` command for the real Megatron path.

Acceptance conditions are config-driven and emitted into ``report.md``:
- ``loss_drop_first_50_steps``: training loss must fall by at least this delta.
- ``must_observe_restart_at_step``: LR at that step must equal max_lr.
- ``loss_at_step_<S>_below``: optional terminal loss bound.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import (  # noqa: E402
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_json,
    write_text,
    write_yaml,
)

MISSION_ID = "l10_megatron_pretrain_lifecycle"


def _load_train_step():
    """Import student's patch first, fall back to reference for fallback mode."""
    sys.path.insert(0, str(LAB_DIR / "patch"))
    try:
        from starter.train_step import TrainStepError, train_step

        impl = "starter"
    except (ImportError, NotImplementedError):
        from reference.train_step import TrainStepError, train_step

        impl = "reference"
    return train_step, TrainStepError, impl


def _load_lr_scheduler_factory():
    """Try to import the student's L04 CosineWithRestartsLR; else use a local fallback."""
    l04_patch = ROOT / "labs" / "l08_megatron_text_pretrain" / "patch"
    if l04_patch.is_dir():
        spec = importlib.util.spec_from_file_location(
            "l04_lr_scheduler", l04_patch / "starter" / "lr_scheduler.py"
        )
        if spec is not None:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                if hasattr(module, "CosineWithRestartsLR"):
                    return module.CosineWithRestartsLR, "l04_starter"
            except Exception:
                pass
        spec = importlib.util.spec_from_file_location(
            "l04_lr_scheduler_ref", l04_patch / "reference" / "lr_scheduler.py"
        )
        if spec is not None:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                if hasattr(module, "CosineWithRestartsLR"):
                    return module.CosineWithRestartsLR, "l04_reference"
            except Exception:
                pass
    return _LocalCosineWithRestarts, "local_fallback"


class _LocalCosineWithRestarts:
    """Tiny fallback so this script runs even when L04 patch is empty."""

    def __init__(self, optimizer, max_lr, min_lr, restart_steps, total_steps):
        self.optimizer = optimizer
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.restart_steps = sorted(int(step) for step in restart_steps)
        self.total_steps = int(total_steps)
        self._step = 0
        self._set(self.max_lr)

    def _segment(self, step: int) -> tuple[int, int]:
        boundaries = [0, *self.restart_steps, self.total_steps]
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= step < boundaries[i + 1]:
                return boundaries[i], boundaries[i + 1]
        return boundaries[-2], boundaries[-1]

    def _lr_at(self, step: int) -> float:
        if step >= self.total_steps:
            return self.min_lr
        seg_start, seg_end = self._segment(step)
        t = (step - seg_start) / max(1, (seg_end - seg_start))
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * t))

    def step(self) -> None:
        self._step += 1
        self._set(self._lr_at(self._step))

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _set(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def _build_model_and_optimizer(model_cfg, opt_cfg):
    """Build a tiny Llama-style decoder so the lifecycle is meaningful on CPU."""
    import torch
    from torch import nn

    class TinyDecoder(nn.Module):
        def __init__(self, vocab, hidden, layers, heads, seq):
            super().__init__()
            self.tok = nn.Embedding(vocab, hidden)
            self.pos = nn.Embedding(seq, hidden)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
                batch_first=True, activation="gelu", norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, ids):
            B, T = ids.shape
            pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
            x = self.tok(ids) + self.pos(pos)
            mask = torch.triu(torch.ones(T, T, device=ids.device), diagonal=1).bool()
            x = self.blocks(x, mask=mask, is_causal=True)
            x = self.norm(x)
            return self.head(x)

    model = TinyDecoder(
        vocab=int(model_cfg["vocab_size"]),
        hidden=int(model_cfg["hidden_size"]),
        layers=int(model_cfg["num_layers"]),
        heads=int(model_cfg["num_attention_heads"]),
        seq=512,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
    )
    return model, optimizer


def _make_data_iterator(seed: int, vocab: int, batch: int, seq: int):
    import torch

    generator = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(0, vocab, (batch, seq), generator=generator)


def _print_real_megatron_command(config: dict, run_dir: Path) -> None:
    template = config.get("megatron_command_template")
    if not template:
        return
    payload = {
        "tp": config["parallel"]["tp"],
        "pp": config["parallel"]["pp"],
        "num_layers": config["model"]["num_layers"],
        "hidden_size": config["model"]["hidden_size"],
        "num_attention_heads": config["model"]["num_attention_heads"],
        "seq_length": config["training"]["seq_length"],
        "micro_batch_size": config["training"]["micro_batch_size"],
        "global_batch_size": config["training"]["global_batch_size"],
        "train_steps": config["training"]["train_steps"],
        "lr": config["optimizer"]["lr"],
        "min_lr": config["optimizer"]["min_lr"],
        "indexed_prefix": config.get("data", {}).get(
            "indexed_prefix", "data/fineweb_edu/indexed/fineweb_text_document"
        ),
        "save_dir": config.get("checkpoint", {}).get("save_dir", "artifacts/ckpt"),
        "save_every": config.get("checkpoint", {}).get("save_every", 500),
    }
    rendered = " ".join(template.format(**payload).split())
    write_text(run_dir / "artifacts" / "real_megatron_command.sh", "#!/usr/bin/env bash\n" + rendered + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_smoke.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_step, TrainStepError, train_step_impl = _load_train_step()
    SchedulerCls, scheduler_impl = _load_lr_scheduler_factory()
    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))

    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {
            "mission": MISSION_ID,
            "train_step_impl": train_step_impl,
            "scheduler_impl": scheduler_impl,
            **config,
        },
    )
    _print_real_megatron_command(config, run_dir)

    try:
        import torch
    except ImportError:
        write_text(run_dir / "artifacts" / "fallback_reason.txt", "torch not installed\n")
        write_text(run_dir / "report.md", _fallback_report(MISSION_ID))
        print(run_dir)
        return

    torch.manual_seed(args.seed)
    model, optimizer = _build_model_and_optimizer(config["model"], config["optimizer"])
    scheduler = SchedulerCls(
        optimizer,
        max_lr=float(config["optimizer"]["lr"]),
        min_lr=float(config["optimizer"]["min_lr"]),
        restart_steps=list(config["scheduler"]["restart_steps"]),
        total_steps=int(config["scheduler"]["total_steps"]),
    )

    vocab = int(config["model"]["vocab_size"])
    seq = int(config["training"]["seq_length"])
    micro = int(config["training"]["micro_batch_size"])
    train_steps = int(config["training"]["train_steps"])
    data_iter = _make_data_iterator(args.seed, vocab, micro, seq)
    loss_fn = torch.nn.CrossEntropyLoss()

    def forward_backward(data_iterator, model_):
        ids = next(data_iterator)
        logits = model_(ids[:, :-1])
        targets = ids[:, 1:]
        loss = loss_fn(logits.reshape(-1, vocab), targets.reshape(-1))
        loss.backward()
        return {"loss": float(loss.detach()), "tokens": int(ids.numel())}

    save_every = int(config.get("checkpoint", {}).get("save_every", 0))
    save_dir = LAB_DIR / config.get("checkpoint", {}).get("save_dir", "artifacts/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    losses: list[float] = []
    lrs: list[float] = []
    restart_lr_observed: dict[int, float] = {}
    must_restart_at = int(config.get("acceptance", {}).get("must_observe_restart_at_step", -1))
    for step in range(1, train_steps + 1):
        metrics = train_step(forward_backward, data_iter, model, optimizer, scheduler, step)
        loss = float(metrics["loss"])
        lr = float(metrics["lr"])
        losses.append(loss)
        lrs.append(lr)
        if step == must_restart_at:
            restart_lr_observed[step] = lr
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "metric_type": "train",
                "iteration": step,
                "loss": loss,
                "lr": lr,
                "num_microbatches": metrics["num_microbatches"],
                "skipped_iter": metrics["skipped_iter"],
                "tokens": metrics.get("tokens"),
            },
        )
        if save_every and step % save_every == 0:
            ckpt_path = save_dir / f"iter_{step:07d}.pt"
            torch.save({"step": step, "loss": loss, "lr": lr}, ckpt_path)
            (save_dir / "latest_checkpointed_iteration.txt").write_text(str(step))

    # Acceptance
    acceptance = config.get("acceptance", {})
    drop = float(acceptance.get("loss_drop_first_50_steps", 0.0))
    head = sum(losses[:5]) / 5
    tail = sum(losses[45:50]) / 5 if len(losses) >= 50 else losses[-1]
    loss_drop_ok = (head - tail) >= drop
    restart_ok = (must_restart_at < 0) or (
        restart_lr_observed.get(must_restart_at, 0.0)
        >= 0.95 * float(config["optimizer"]["lr"])
    )
    final_loss_ok = True
    final_loss_bound = None
    for key, value in acceptance.items():
        if key.startswith("loss_at_step_") and key.endswith("_below"):
            final_loss_bound = float(value)
            final_loss_ok = losses[-1] < final_loss_bound

    write_json(
        run_dir / "artifacts" / "acceptance.json",
        {
            "head_loss": head,
            "tail_loss_at_50": tail,
            "loss_drop": head - tail,
            "loss_drop_required": drop,
            "loss_drop_ok": loss_drop_ok,
            "must_restart_at": must_restart_at,
            "restart_lr_observed": restart_lr_observed,
            "restart_ok": restart_ok,
            "final_loss": losses[-1],
            "final_loss_bound": final_loss_bound,
            "final_loss_ok": final_loss_ok,
            "all_passed": loss_drop_ok and restart_ok and final_loss_ok,
        },
    )

    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
把 patch 的 `train_step` 接回真实预训练生命周期，并产出 loss 下降证据。

## 2. 环境与配置
- profile: {config.get('profile')}
- train_step impl: {train_step_impl}
- scheduler impl: {scheduler_impl}
- model: hidden={config['model']['hidden_size']}, layers={config['model']['num_layers']}

## 3. 预测
- loss 应在前 50 步下降 ≥ {drop}
- LR 应在 step={must_restart_at} 重启回 max_lr
- 终态 loss 应 < {final_loss_bound}

## 4. 运行命令
见 `command.sh`。如果配置带 `megatron_command_template`，真实 Megatron 命令在
`artifacts/real_megatron_command.sh`。

## 5. 结果
- 头部 5 步均 loss: {head:.4f}
- 第 50 步附近 loss: {tail:.4f}（drop={head - tail:.4f}）
- restart 处 LR: {restart_lr_observed}
- 终态 loss: {losses[-1]:.4f}
- 全部通过： {loss_drop_ok and restart_ok and final_loss_ok}

## 6. 诊断
若 loss 不下降，先检查 train_step 是否在 forward_backward 之前 zero_grad；
若 LR 没有重启，回到 L04 检查 `CosineWithRestartsLR.restart_steps` 处理；
若 step 被频繁 skip，看 grad_norm 是否爆掉。

## 7. Debug 工单
推荐 `mgt_lifecycle_train_loss_nan`、`mgt_lifecycle_skip_step_storm`。

## 8. PR Review
评审 patch 时确认：
- zero_grad 必须在 forward_backward 之前
- scheduler.step() 必须在 optimizer 成功后
- 多 microbatch loss 必须取平均

## 9. 我原来误解了什么
（学习者自填）

## 10. 如果迁移到 8×H200
切换到 `configs/h200_125m.yaml`，按 `artifacts/real_megatron_command.sh` 启动
`pretrain_gpt.py`，把 metrics.jsonl 的 schema 保持一致即可。

## 11. 下一步
进入 L05 学 bucketed DDP，用同一个 train_step 串到 multi-rank 训练。
""",
    )
    print(run_dir)


def _fallback_report(mission_id: str) -> str:
    return f"""# Mission Report：{mission_id}

torch 未安装，本次 run 走了 fallback 路径，仅生成 config.resolved.yaml 与
real_megatron_command.sh（如果配置提供了模板）。请在有 torch 的环境重新跑
`scripts/run_lifecycle.py` 才能产出 loss 证据。
"""


if __name__ == "__main__":
    main()
