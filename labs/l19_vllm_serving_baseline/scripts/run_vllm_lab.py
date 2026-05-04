from __future__ import annotations

import argparse
import importlib.util
import json
import socket
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
    append_jsonl,
    ensure_prediction,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_text,
    write_yaml,
)

MISSION_ID = "l19_vllm_serving_baseline"
LAB_DIR = Path(__file__).resolve().parents[1]


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--config", default="configs/4090_debug.yaml")
    parser.add_argument("--mode", default="smoke")
    args = parser.parse_args()
    cfg = yaml.safe_load((LAB_DIR / args.config).read_text(encoding="utf-8"))
    run_dir = prepare_run_dir(MISSION_ID, args.run_id, ROOT)
    write_command_snapshot(run_dir)
    ensure_prediction(run_dir / "prediction.yaml")
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": MISSION_ID, "mode": args.mode, **cfg},
    )
    available = importlib.util.find_spec("vllm") is not None
    server = port_open(int(cfg["port"]))
    command = f"vllm serve {cfg['model']} --port {cfg['port']} --max-model-len 4096"
    write_text(
        run_dir / "serve.log",
        f"[{utc_now()}] vLLM 服务验证\nvllm_available={available}\nserver_port_open={server}\ncommand={command}\n",
    )
    ttft = None
    ok = False
    if server:
        req = urllib.request.Request(
            f"http://127.0.0.1:{cfg['port']}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": cfg["model"],
                    "messages": [{"role": "user", "content": "用一句话解释 TTFT"}],
                    "max_tokens": cfg["max_new_tokens"],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            urllib.request.urlopen(req, timeout=30).read()
            ttft = (time.perf_counter() - start) * 1000
            ok = True
        except urllib.error.URLError:
            ok = False
    row = {
        "timestamp": utc_now(),
        "metric_type": "serve",
        "server_available": server,
        "vllm_available": available,
        "requests_per_sec": round(1000 / max(ttft or 1000, 1), 3) if ok else 0.0,
        "ttft_ms_p50": round(ttft, 2) if ttft else None,
        "itl_ms_p50": None,
        "output_tokens_per_sec": 0.0 if not ok else None,
        "status": "served" if ok else "validation_only",
    }
    append_jsonl(run_dir / "metrics.jsonl", row)
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
建立 vLLM OpenAI-compatible serving baseline。

## 2. 环境与配置
- model：{cfg['model']}
- port：{cfg['port']}
- vLLM installed：{available}
- server available：{server}

## 3. 预测
服务端最常见问题是端口冲突、模型加载失败和 KV cache 显存压力。

## 4. 运行命令
`{command}`，详见 `command.sh`。

## 5. 结果
- status：{row['status']}
- TTFT p50：{row['ttft_ms_p50']}

## 6. 诊断
如果是 validation_only，说明本地没有正在运行的 vLLM server；报告不会把它伪装成真实 serving 结果。

## 7. Debug Ticket
建议练习 `vllm_memory_pressure_001` 或 `vllm_port_conflict_003`。

## 8. PR Review
修改 chat template 或模型名时必须重跑客户端请求。

## 9. 我原来误解了什么

## 10. 如果迁移到 8×H200
增加并发 sweep，记录 TTFT、ITL、吞吐和显存。

## 11. 下一步
用同一 workload 对比 SGLang。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
