from __future__ import annotations

import argparse
import importlib.util
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import (
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_json,
    write_text,
    write_yaml,
)

MISSION_ID = "l06_torchtitan_training"
LAB_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TorchTitan 教学 smoke：训练框架边界验证"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--config", default="configs/4090_debug.toml")
    parser.add_argument("--mode", default="smoke")
    args = parser.parse_args()

    config = tomllib.loads((LAB_DIR / args.config).read_text(encoding="utf-8"))
    steps = int(config.get("training", {}).get("steps", 10))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "mode": args.mode, **config},
    )

    import torch
    from torch import nn

    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(steps):
        x = torch.randn(16, 8, device=device)
        target = torch.randn(16, 4, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "step": step + 1,
                "metric_type": "train",
                "loss": losses[-1],
                "tokens_per_sec": 16 * 8,
            },
        )

    ckpt = run_dir / "artifacts" / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": steps,
        },
        ckpt,
    )
    reloaded = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)).to(device)
    reloaded.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    with torch.no_grad():
        resume_delta = float(
            (
                reloaded(torch.zeros(1, 8, device=device))
                - model(torch.zeros(1, 8, device=device))
            )
            .abs()
            .max()
            .cpu()
        )

    torchtitan_available = importlib.util.find_spec("torchtitan") is not None
    write_json(
        run_dir / "artifacts" / "framework_validation.json",
        {
            "torchtitan_available": torchtitan_available,
            "device": str(device),
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "resume_delta": resume_delta,
        },
    )
    write_text(
        run_dir / "train.log",
        f"[{utc_now()}] TorchTitan 边界 smoke 完成；torchtitan_available={torchtitan_available}; checkpoint={ckpt}\n",
    )
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
验证训练框架应负责的配置、训练循环、checkpoint、resume 与 metrics 边界。

## 2. 环境与配置
- 配置：`{args.config}`
- TorchTitan 是否安装：{torchtitan_available}
- 设备：{device}

## 3. 预测
最容易失败的是配置字段漂移和 checkpoint 恢复语义不一致。

## 4. 运行命令
见 `command.sh`。

## 5. 结果
- loss_start：{losses[0]:.6f}
- loss_end：{losses[-1]:.6f}
- checkpoint：`{ckpt.relative_to(ROOT)}`
- resume_delta：{resume_delta:.6g}

## 6. 诊断
本地 smoke 不伪装成完整 TorchTitan 集群训练；它验证训练框架边界。若需要真实 TorchTitan，请按官方仓库命令在 H200 环境运行。

## 7. Debug Ticket
建议练习 `tt_checkpoint_002`。

## 8. PR Review
修改 FSDP/TP/checkpoint 配置必须验证 resume，而不只是保存文件。

## 9. 我原来误解了什么

## 10. 如果迁移到 8×H200
使用 TorchTitan 官方配置启动 FSDP2/TP，对比 checkpoint 大小、resume 时间与显存。

## 11. 下一步
进入 Megatron 前，比较 TorchTitan 与 raw PyTorch 的配置所有权。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
