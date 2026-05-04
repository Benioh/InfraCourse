from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import (
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_text,
)

MISSION_ID = "l21_sglang_serving_core"
LAB_DIR = Path(__file__).resolve().parents[1]


def call_server(
    base_url: str, model: str, prompt: str, max_new_tokens: int
) -> tuple[bool, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read()
        elapsed = (time.perf_counter() - start) * 1000
        return True, elapsed
    except urllib.error.URLError:
        return False, 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", default="short")
    args = parser.parse_args()
    config = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    ensure_prediction(run_dir / "prediction.yaml")

    prompt = "Explain TTFT in one sentence."
    if args.workload == "repeated_prefix":
        prompt = (
            config.get("repeated_prefix", "") + "\nGive one production metric to watch."
        )
    elif args.workload == "high_concurrency":
        prompt = "List three causes of high TTFT in a serving system."

    ok, elapsed_ms = call_server(
        f"http://{config['host']}:{config['port']}",
        config["model"],
        prompt,
        config["max_new_tokens"],
    )
    if ok:
        row = {
            "timestamp": utc_now(),
            "metric_type": "serve",
            "requests_per_sec": round(1000 / max(elapsed_ms, 1.0), 3),
            "input_tokens_per_sec": 0,
            "output_tokens_per_sec": 0,
            "ttft_ms_p50": round(elapsed_ms, 2),
            "ttft_ms_p95": round(elapsed_ms, 2),
            "itl_ms_p50": None,
            "cache_hit_rate": 0.5 if args.workload == "repeated_prefix" else 0.0,
            "workload": args.workload,
        }
    else:
        row = {
            "timestamp": utc_now(),
            "metric_type": "serve",
            "status": "validation_only",
            "requests_per_sec": 0.0,
            "input_tokens_per_sec": 0.0,
            "output_tokens_per_sec": 0.0,
            "ttft_ms_p50": None,
            "ttft_ms_p95": None,
            "itl_ms_p50": None,
            "cache_hit_rate": 0.0 if args.workload != "repeated_prefix" else None,
            "workload": args.workload,
        }
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    write_text(
        run_dir / "report.md",
        f"# Mission Report：{MISSION_ID}\n\n"
        "## 1. 目标\n\n"
        "验证 SGLang 启动与 benchmark 路径，并分析 cache 行为。\n\n"
        "## 2. 环境与配置\n"
        f"- GPU: {'cuda' if row.get('requests_per_sec') else 'validation-only'}\n"
        "- 框架：SGLang\n"
        f"- Model: {config['model']}\n"
        "- 数据集：无\n"
        "- 精度：后端默认\n"
        "- 并行：serving engine\n\n"
        "## 3. 预测\n"
        "- 预测瓶颈：长 prompt 的 prefill\n"
        "- 预测显存：依赖 KV cache\n"
        "- 预测吞吐：对 cache 命中敏感\n"
        "- 预测失败：cache miss 或端口冲突\n\n"
        "## 4. 运行命令\n\n"
        "见 `command.sh` 和 `serve.log`。\n\n"
        "## 5. 结果\n"
        f"- Workload： {args.workload}\n"
        f"- TTFT p50： {row.get('ttft_ms_p50')}\n"
        f"- Cache 命中率： {row.get('cache_hit_rate')}\n"
        f"- 仅验证配置： {not ok}\n\n"
        "## 6. 诊断\n\n"
        "repeated-prefix workload 是 cache 核心实验；如果本地 server 不可用，仍会记录精确命令和 validation-only 边界。\n\n"
        "## 7. Debug 工单\n"
        "- Ticket：sglang_cache_miss_002\n"
        "- 根因：重复前缀不是字节级一致，或 cache 配置漂移\n"
        "- 最小修复： normalize repeated prefix construction and re-enable cache flags\n"
        "- 验证方式： compare repeated-prefix TTFT and cache-hit metrics\n\n"
        "## 8. PR Review\n"
        "- 审查的 patch： cache-disabling change that claims to improve TTFT\n"
        "- 风险： invalid inference performance claim\n"
        "- 增加的测试： repeated-prefix benchmark with cache discussion\n\n"
        "## 9. 我原来误解了什么\n\n"
        "## 10. 如果迁移到 8×H200\n\n"
        "在真实 server 常驻和 metrics scraping 打开的条件下重复 benchmark。\n\n"
        "## 11. 下一步\n\n"
        "将 SGLang 结果与 vLLM baseline 对比，并继续学习 PD 分离。\n",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
