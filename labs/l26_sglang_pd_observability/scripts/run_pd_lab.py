from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import yaml

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

MISSION_ID = "l26_sglang_pd_observability"
LAB_DIR = Path(__file__).resolve().parents[1]


def simulate(name, prefill, decode):
    total = max(prefill + decode, 1)
    balance = abs(prefill - decode) / total
    return {
        "config": name,
        "prefill_gpus": prefill,
        "decode_gpus": decode,
        "ttft_ms_p50": round(900 / max(prefill, 1) + 40, 2),
        "itl_ms_p50": round(80 / max(decode, 1) + 3, 2),
        "queue_wait_ms": round(30 + 120 * balance, 2),
        "routing_errors": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--mode", default="pd")
    args = parser.parse_args()
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml", {"mission": MISSION_ID, "mode": args.mode}
    )
    rows = []
    for cfg_name in ["unified_tp8.yaml", "pd_2p6d.yaml", "pd_4p4d.yaml"]:
        cfg = yaml.safe_load(
            (LAB_DIR / "configs" / cfg_name).read_text(encoding="utf-8")
        )
        if cfg.get("type") == "unified":
            row = simulate("unified_tp8", cfg["tp"] // 2, cfg["tp"] // 2)
        else:
            row = simulate(
                cfg_name.replace(".yaml", ""),
                int(cfg["prefill_gpus"]),
                int(cfg["decode_gpus"]),
            )
        rows.append(row)
        append_jsonl(
            run_dir / "metrics.jsonl",
            {"timestamp": utc_now(), "metric_type": "serve", **row},
        )
    write_json(run_dir / "artifacts" / "pd_comparison.json", rows)
    write_text(
        run_dir / "serve.log",
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
    )
    best_ttft = min(rows, key=lambda r: r["ttft_ms_p50"])
    best_itl = min(rows, key=lambda r: r["itl_ms_p50"])
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
比较 unified 与 PD 分离在长 prefill/long decode 下的延迟形态。

## 2. 环境与配置
- configs：unified_tp8、pd_2p6d、pd_4p4d

## 3. 预测
prefill GPU 更多会降低 TTFT，decode GPU 更多会降低 ITL。

## 4. 运行命令
见 `command.sh`。

## 5. 结果
- 最低 TTFT：{best_ttft}
- 最低 ITL：{best_itl}

## 6. 诊断
PD 分离不是无条件更好；资源分配必须匹配 workload。

## 7. Debug Ticket
建议练习 `sglang_pd_misroute_003` 和 `sglang_pd_decode_starve_004`。

## 8. PR Review
修改 router 规则必须保留 engine 类型指标，否则无法判断 misroute。

## 9. 我原来误解了什么

## 10. 如果迁移到 8×H200
替换模拟指标为真实 SGLang metrics endpoint，并保留相同字段。

## 11. 下一步
加入混合短请求/长请求 workload。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
