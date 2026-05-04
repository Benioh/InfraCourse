"""L05.3 · 在多 GPU 上跑 FSDP2 wrap + 训练 smoke。

CPU dryrun: 只跑 wrap_only 模式，验证 wrapped_blocks 数量。
GPU train_smoke: 用 torchrun 启动，初始化 NCCL 进程组，wrap Llama-style
模型并跑前向 + 反向 + optimizer step。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB_DIR / "patch"))

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

MISSION_ID = "l12_fsdp2_llama"


def _impl():
    try:
        from starter import fsdp2_wrap as mod  # type: ignore[import-not-found]

        return mod, "starter"
    except (ImportError, NotImplementedError):
        from reference import fsdp2_wrap as mod  # type: ignore[import-not-found]

        return mod, "reference"


def _build_model(model_cfg):
    import torch
    from torch import nn

    class _Block(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
            )
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.mlp(x))
            return x

    class _Llama(nn.Module):
        def __init__(self, vocab, dim, depth, heads, seq):
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab, bias=False)
            self.seq = seq

        def forward(self, ids):
            x = self.embed(ids)
            for block in self.blocks:
                x = block(x)
            return self.head(self.norm(x))

    model = _Llama(
        vocab=int(model_cfg["vocab_size"]),
        dim=int(model_cfg["hidden_size"]),
        depth=int(model_cfg["num_layers"]),
        heads=int(model_cfg["num_attention_heads"]),
        seq=512,
    )
    return model, _Block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu_dryrun.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    mod, impl_label = _impl()
    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "impl": impl_label, **config},
    )

    if "torchrun_command" in config:
        write_text(
            run_dir / "artifacts" / "torchrun_command.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + " ".join(config["torchrun_command"].split())
            + "\n",
        )

    try:
        import torch
        from torch import nn  # noqa: F401
    except ImportError:
        write_text(run_dir / "artifacts" / "fallback.txt", "torch not installed\n")
        write_text(run_dir / "report.md", _fallback_report(MISSION_ID))
        print(run_dir)
        return

    model, BlockCls = _build_model(config["model"])
    mode = config.get("mode", "wrap_only")

    captured: list[tuple[str, dict]] = []

    def spy_fully_shard(module, **kwargs):
        captured.append((module.__class__.__name__, kwargs))
        return module

    if mode == "wrap_only":
        report = mod.wrap_transformer_blocks_fsdp2(
            model,
            BlockCls,
            mp_policy=None,
            reshard_after_forward=bool(config.get("reshard_after_forward", True)),
            _fully_shard=spy_fully_shard,
        )
        accepted = (
            len(report.wrapped_blocks)
            == int(config.get("acceptance", {}).get("expect_wrapped_blocks", -1))
        )
        write_json(
            run_dir / "artifacts" / "wrap_report.json",
            {
                "wrapped_blocks": report.wrapped_blocks,
                "root_wrapped": report.root_wrapped,
                "mp_policy_summary": report.mp_policy_summary,
                "accepted": accepted,
            },
        )
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "metric_type": "wrap",
                "wrapped_blocks": len(report.wrapped_blocks),
                "accepted": accepted,
            },
        )
        write_text(run_dir / "report.md", _wrap_report_md(MISSION_ID, impl_label, config, report, accepted))
        print(run_dir)
        return

    # train_smoke
    if not torch.cuda.is_available():
        write_text(run_dir / "artifacts" / "fallback.txt", "no CUDA; train_smoke skipped\n")
        write_text(run_dir / "report.md", _fallback_report(MISSION_ID))
        print(run_dir)
        return

    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        dist.init_process_group("nccl", rank=0, world_size=int(os.environ.get("WORLD_SIZE", "1")))
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda")
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

    model = model.to(device)
    policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, config["mp_policy"]["param_dtype"]),
        reduce_dtype=getattr(torch, config["mp_policy"]["reduce_dtype"]),
    )
    mod.wrap_transformer_blocks_fsdp2(
        model,
        BlockCls,
        mp_policy=policy,
        reshard_after_forward=bool(config.get("reshard_after_forward", True)),
        _fully_shard=fully_shard,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["optimizer"]["lr"]))
    train_steps = int(config["training"]["train_steps"])
    seq = int(config["training"]["seq_length"])
    micro = int(config["training"]["micro_batch_size"])
    vocab = int(config["model"]["vocab_size"])
    losses: list[float] = []
    peak_gb = 0.0
    start = time.perf_counter()
    for step in range(1, train_steps + 1):
        ids = torch.randint(0, vocab, (micro, seq), device=device)
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach()))
        peak_gb = max(peak_gb, torch.cuda.max_memory_allocated() / 1024**3)
        if rank == 0:
            append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "timestamp": utc_now(),
                    "metric_type": "train",
                    "step": step,
                    "loss": losses[-1],
                    "peak_memory_gb": peak_gb,
                },
            )

    duration = time.perf_counter() - start
    if rank == 0:
        head = sum(losses[:5]) / 5
        tail_window = (
            sum(losses[45:50]) / 5 if len(losses) >= 50 else losses[-1]
        )
        loss_drop = head - tail_window
        accept = config.get("acceptance", {})
        loss_drop_required = float(
            accept.get("loss_drop_first_50_steps", accept.get("loss_drop_first_25_steps", 0.0))
        )
        peak_under = float(accept.get("peak_memory_gb_under", 1e9))
        all_pass = loss_drop >= loss_drop_required and peak_gb <= peak_under
        write_json(
            run_dir / "artifacts" / "train_smoke.json",
            {
                "loss_head": head,
                "loss_tail": tail_window,
                "loss_drop": loss_drop,
                "peak_memory_gb": peak_gb,
                "duration_s": duration,
                "all_pass": all_pass,
            },
        )
        write_text(
            run_dir / "report.md",
            _train_report_md(MISSION_ID, impl_label, config, head, tail_window, loss_drop, peak_gb, all_pass),
        )
    if dist.is_initialized():
        dist.barrier()
    print(run_dir)


def _fallback_report(mission_id):
    return f"# Mission Report：{mission_id}\n\ntorch / CUDA 不可用，仅生成命令模板。\n"


def _wrap_report_md(mission, impl, config, report, accepted):
    return f"""# Mission Report：{mission}

## 1. 目标
仅验证 wrap 行为：每个 block 都被 fully_shard 包过，root 最后一个。

## 2. 配置
profile={config.get('profile')}, impl={impl}

## 3. 结果
- wrapped_blocks: {report.wrapped_blocks}
- root_wrapped: {report.root_wrapped}
- mp_policy: {report.mp_policy_summary}
- accepted: {accepted}

## 4. 下一步
切换到 `configs/h200_llama1b.yaml`，按 `artifacts/torchrun_command.sh` 启动多 GPU 训练。
"""


def _train_report_md(mission, impl, config, head, tail, drop, peak, ok):
    return f"""# Mission Report：{mission}

## 1. 目标
在 {config['model'].get('size_label', '?')} 模型上跑 FSDP2 + AdamW 训练 smoke。

## 2. 配置
- impl: {impl}
- mp_policy: {config['mp_policy']}
- reshard_after_forward: {config.get('reshard_after_forward')}
- micro_bs / seq: {config['training']['micro_batch_size']} / {config['training']['seq_length']}

## 3. 结果
- 头部 loss: {head:.4f}
- 末段 loss: {tail:.4f}
- loss drop: {drop:.4f}
- peak memory (GB): {peak:.2f}
- acceptance ok: {ok}

## 4. 诊断
- 如果显存爆：把 `reshard_after_forward` 设 True 或缩小 seq_length
- 如果 loss 不动：检查是否所有 block 都被 wrap、是否 mp_policy reduce_dtype 是 fp32

## 5. 下一步
进入 L05.5 看 MoE / EP，然后回到 L04.8 用 FSDP2 替换 ManualDDP。
"""


if __name__ == "__main__":
    main()
