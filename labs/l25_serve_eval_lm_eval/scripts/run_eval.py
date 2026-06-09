"""L26 · 用 patch 的 harness 跑 GSM8K-style 评测。

stub 模式：不需要任何外部服务。
openai 模式：通过任意 OpenAI-compatible endpoint 打分（vLLM / SGLang / OpenAI）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib import request as urlreq

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

MISSION_ID = "l25_serve_eval_lm_eval"


def _impl():
    import importlib
    import os

    requested = os.environ.get("IMPL", "starter")
    try:
        return importlib.import_module(f"{requested}.eval_harness"), requested
    except (ImportError, NotImplementedError):
        return importlib.import_module("reference.eval_harness"), "reference"


def _toy_gsm8k(n: int) -> list[dict]:
    items = []
    for i in range(n):
        a = (i * 7 + 3) % 100
        b = (i * 11 + 5) % 50
        items.append(
            {
                "question": f"What is {a} plus {b}?",
                "answer": f"{a + b}",
            }
        )
    return items


class StubClient:
    def complete(self, prompt: str, max_tokens: int = 256, stop=None):
        # answer the math by reading the last "What is X plus Y" pattern from prompt
        import re

        match = re.findall(r"What is (\d+) plus (\d+)\?", prompt)
        if match:
            a, b = match[-1]
            return f" {int(a) + int(b)}"
        return " 0"


class OpenAIClient:
    def __init__(self, endpoint: str, model: str, api_key: str = "EMPTY"):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key

    def complete(self, prompt: str, max_tokens: int = 256, stop=None):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if stop:
            payload["stop"] = stop
        req = urlreq.Request(
            f"{self.endpoint}/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urlreq.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0].get("text", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_stub.yaml")
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

    if "prerequisite_command" in config:
        write_text(
            run_dir / "artifacts" / "prerequisite_serve_command.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + " ".join(config["prerequisite_command"].split())
            + "\n",
        )

    n_eval = int(config["n_eval"])
    n_shots = int(config["n_shots"])
    items = _toy_gsm8k(n_eval + n_shots)

    if config.get("mode", "stub") == "stub":
        client = StubClient()
    else:
        client = OpenAIClient(config["endpoint"], config["model"])

    start = time.perf_counter()
    try:
        result = mod.run_evaluation(items, client, n_shots=n_shots)
        ok = True
    except Exception as exc:  # noqa: BLE001
        write_text(run_dir / "artifacts" / "fallback.txt", f"eval failed: {exc}\n")
        result = {"exact_match": 0.0, "first_number_match": 0.0, "n_total": 0, "samples": []}
        ok = False
    duration = time.perf_counter() - start

    accept = config.get("acceptance", {})
    must_complete = accept.get("must_complete", True)
    em_min = float(accept.get("exact_match_min", 0.0))
    fnm_min = float(accept.get("first_number_match_min", 0.0))
    accept_ok = (
        ok
        and (not must_complete or result["n_total"] == n_eval)
        and result["exact_match"] >= em_min
        and result["first_number_match"] >= fnm_min
    )

    for sample in result["samples"]:
        append_jsonl(
            run_dir / "metrics.jsonl",
            {
                "timestamp": utc_now(),
                "metric_type": "eval_sample",
                **sample,
            },
        )
    write_json(
        run_dir / "artifacts" / "eval_summary.json",
        {
            "impl": impl_label,
            "exact_match": result["exact_match"],
            "first_number_match": result["first_number_match"],
            "n_total": result["n_total"],
            "duration_s": duration,
            "accept": accept_ok,
        },
    )
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{MISSION_ID}

## 1. 目标
对 GSM8K-style 数据跑评测 harness。

## 2. 配置
- profile: {config.get('profile')}
- impl: {impl_label}
- mode: {config.get('mode')}
- n_shots / n_eval: {n_shots} / {n_eval}

## 3. 结果
- exact_match: {result['exact_match']:.3f}
- first_number_match: {result['first_number_match']:.3f}
- n_total: {result['n_total']}
- duration: {duration:.2f}s
- accept: {accept_ok}

## 4. 命令
见 `command.sh`；profile 含 `prerequisite_command` 时另见 `artifacts/prerequisite_serve_command.sh`。

## 5. 下一步
真实评测：先启 vLLM/SGLang，再用 `--config configs/local_vllm_qwen.yaml`。
""",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
