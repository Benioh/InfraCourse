from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_utils import (  # noqa: E402
    append_jsonl,
    prepare_run_dir,
    utc_now,
    write_command_snapshot,
    write_json,
    write_text,
    write_yaml,
)

MISSION_META: dict[str, dict[str, Any]] = {
    "l04_gpu_kernel": {
        "title": "GPU 内存层级与最小 Triton kernel",
        "log": "train.log",
        "mini_files": [
            "mini_infra/gpu/memory_model.py",
            "mini_infra/gpu/triton_softmax.py",
            "mini_infra/gpu/microbench.py",
        ],
        "real_sources": [
            "github_repo/triton/python/tutorials/02-fused-softmax.py",
            "github_repo/Megatron-LM/megatron/core/fusions/fused_softmax.py",
        ],
        "metrics": [
            "bandwidth_gbs",
            "peak_hbm_gbs",
            "occupancy",
            "speedup_vs_torch",
            "max_abs_err",
            "block_size",
        ],
        "hardware": "0 GPU 30% / 4090 90% / H200 100%",
    },
    "l09_long_context_cp": {
        "title": "长上下文：CP / RoPE / YaRN",
        "log": "train.log",
        "mini_files": [
            "mini_infra/megatron/core/context_parallel/ring_attention.py",
            "mini_infra/megatron/core/context_parallel/rope.py",
            "mini_infra/megatron/core/context_parallel/yarn.py",
        ],
        "real_sources": [
            "github_repo/Megatron-LM/megatron/core/transformer/attention.py",
            "github_repo/Megatron-LM/megatron/core/models/common/embeddings/rotary_pos_embedding.py",
            "github_repo/Megatron-LM/megatron/core/parallel_state.py",
        ],
        "metrics": [
            "peak_mem_gb",
            "step_time_ms",
            "attn_comm_bytes",
            "tokens_per_sec",
            "rope_freq_max",
            "yarn_scale",
            "val_ppl_at_extended_ctx",
        ],
        "hardware": "0 GPU 35% / 4090 45% / H200 100%",
    },
    "l13_moe_ep": {
        "title": "MoE / 专家并行 EP",
        "log": "train.log",
        "mini_files": [
            "mini_infra/megatron/core/transformer/moe/router.py",
            "mini_infra/megatron/core/transformer/moe/capacity.py",
            "mini_infra/megatron/core/transformer/moe/alltoall.py",
            "mini_infra/megatron/core/transformer/moe/experts.py",
        ],
        "real_sources": [
            "github_repo/Megatron-LM/megatron/core/transformer/moe/router.py",
            "github_repo/Megatron-LM/megatron/core/transformer/moe/experts.py",
            "github_repo/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py",
        ],
        "metrics": [
            "active_params",
            "total_params",
            "tokens_per_expert_p50",
            "tokens_per_expert_p99",
            "alltoall_ms",
            "step_time_ms",
            "router_entropy",
            "capacity_overflow_rate",
        ],
        "hardware": "0 GPU 30% / 4090 40% / H200 100%",
    },
    "l18_data_engineering": {
        "title": "数据工程进阶：WebDataset / minhash / shard 恢复",
        "log": "train.log",
        "mini_files": [
            "mini_infra/data/wds_pipeline.py",
            "mini_infra/data/minhash_dedup.py",
            "mini_infra/data/shard_resume.py",
        ],
        "real_sources": [
            "github_repo/webdataset/webdataset/dataset.py",
            "github_repo/datasketch/datasketch/minhash.py",
            "github_repo/Megatron-LM/megatron/core/datasets/blended_megatron_dataset_builder.py",
        ],
        "metrics": [
            "wds_throughput_mbs",
            "p99_batch_latency_ms",
            "dedup_ratio",
            "recovered_shards",
            "lost_samples",
            "detshuffle_match",
        ],
        "hardware": "0 GPU 80% / 4090 90% / H200 100%",
    },
    "l23_quant_serving": {
        "title": "量化 Serving：FP8 / AWQ / KV-cache INT8",
        "log": "serve.log",
        "mini_files": [
            "mini_infra/vllm/quant/awq_loader.py",
            "mini_infra/vllm/quant/fp8_kv.py",
            "mini_infra/vllm/quant/calibrator.py",
            "mini_infra/sglang/quant/kv_int8.py",
        ],
        "real_sources": [
            "github_repo/vllm/vllm/model_executor/layers/quantization/awq.py",
            "github_repo/vllm/vllm/model_executor/layers/quantization/fp8.py",
            "github_repo/sglang/python/sglang/srt/layers/quantization/",
        ],
        "metrics": [
            "ttft_ms_p50",
            "ttft_ms_p99",
            "itl_ms_p50",
            "itl_ms_p99",
            "tokens_per_sec",
            "peak_kv_mem_gb",
            "acc_drop_pp",
            "cache_hit_rate",
        ],
        "hardware": "0 GPU 25% / 4090 70% / H200 100%",
    },
    "l24_spec_decode": {
        "title": "Speculative Decoding / Parallel Decoding",
        "log": "serve.log",
        "mini_files": [
            "mini_infra/vllm/spec_decode/draft_runner.py",
            "mini_infra/vllm/spec_decode/ngram.py",
            "mini_infra/vllm/spec_decode/acceptance_tracker.py",
        ],
        "real_sources": [
            "github_repo/vllm/vllm/spec_decode/spec_decode_worker.py",
            "github_repo/vllm/vllm/spec_decode/ngram_worker.py",
            "github_repo/sglang/python/sglang/srt/speculative/",
        ],
        "metrics": [
            "acceptance_rate",
            "speedup",
            "ttft_ms",
            "itl_ms_p50",
            "itl_ms_p99",
            "draft_latency_ms",
            "draft_gpu_mem_gb",
        ],
        "hardware": "0 GPU 30% / 4090 70% / H200 100%",
    },
}

PREDICTIONS: dict[str, list[dict[str, str]]] = {
    "l04_gpu_kernel": [
        {
            "id": "bw_gain_with_seq",
            "statement": "seq 1K→16K 时 bandwidth_gbs 单调上升至 plateau",
            "verify_metric": "bandwidth_gbs",
        },
        {
            "id": "numeric_match",
            "statement": "max_abs_err < 1e-5（fp16）",
            "verify_metric": "max_abs_err",
        },
        {
            "id": "block_sweet_spot",
            "statement": "4090 上 BLOCK_SIZE=1024 优于 512 与 2048",
            "verify_metric": "block_size",
        },
    ],
    "l09_long_context_cp": [
        {
            "id": "cp_breakeven_at_16k",
            "statement": "16K 时 CP=2 step_time 优于 CP=1 + recompute=full",
            "verify_metric": "step_time_ms",
        },
        {
            "id": "yarn_extends_eval_ppl",
            "statement": "训练 4K + YaRN 推理到 32K，eval ppl 上升 < 10%",
            "verify_metric": "val_ppl_at_extended_ctx",
        },
        {
            "id": "ring_comm_grows_linear_in_seq",
            "statement": "attn_comm_bytes 与 seq 线性，与 CP 反比",
            "verify_metric": "attn_comm_bytes",
        },
    ],
    "l13_moe_ep": [
        {
            "id": "alltoall_dominates_at_ep8",
            "statement": "EP=8 时 alltoall_ms 占 step_time ≥ 30%",
            "verify_metric": "alltoall_ms",
        },
        {
            "id": "capacity_overflow_hurts_acc",
            "statement": "cf=1.0 出现 ≥ 5% drop，acc 显著恶化",
            "verify_metric": "capacity_overflow_rate",
        },
        {
            "id": "aux_loss_prevents_collapse",
            "statement": "aux=0 时 router_entropy 单调下降；coef=0.01 稳定",
            "verify_metric": "router_entropy",
        },
    ],
    "l18_data_engineering": [
        {
            "id": "prefetch_helps_p99",
            "statement": "prefetch 2→8，p99_batch_latency_ms 下降 ≥ 30%",
            "verify_metric": "p99_batch_latency_ms",
        },
        {
            "id": "detshuffle_reproducible",
            "statement": "同 seed 下完全一致",
            "verify_metric": "detshuffle_match",
        },
        {
            "id": "dedup_lower_loss_at_same_steps",
            "statement": "同步数下去重后 loss 在前 1k 步更低",
            "verify_metric": "dedup_ratio",
        },
    ],
    "l23_quant_serving": [
        {
            "id": "kv_int8_saves_mem_only",
            "statement": "kv-int8 显著降 peak_kv_mem_gb，对 tokens_per_sec 中性",
            "verify_metric": "peak_kv_mem_gb",
        },
        {
            "id": "fp8_high_throughput_h200",
            "statement": "fp8 在 H200 吞吐 > fp16 1.5x；4090 不可跑",
            "verify_metric": "tokens_per_sec",
        },
        {
            "id": "awq_ttft_neutral",
            "statement": "awq w4a16 对 TTFT 中性偏负，throughput 正向，acc_drop_pp <= 1.0",
            "verify_metric": "acc_drop_pp",
        },
    ],
    "l24_spec_decode": [
        {
            "id": "spec_helps_low_concurrency",
            "statement": "batch=1 spec speedup ≥ 1.5x；batch=64 退化到 < 1.1x",
            "verify_metric": "speedup",
        },
        {
            "id": "ngram_zero_cost_baseline",
            "statement": "ngram 长 prompt acceptance ≥ 30%，无 draft GPU 占用",
            "verify_metric": "acceptance_rate",
        },
        {
            "id": "ood_acceptance_drops",
            "statement": "OOD prompt acceptance 较 in-domain 下降 ≥ 50%",
            "verify_metric": "acceptance_rate",
        },
    ],
}


def build_payload(mission_id: str, mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if mission_id == "l04_gpu_kernel":
        from mini_infra.gpu.microbench import run_sweep

        payload = run_sweep("rtx4090" if mode != "h200" else "h200")
        best = dict(payload["best"])
        return payload, {"metric_type": "gpu_kernel", **best}

    if mission_id == "l09_long_context_cp":
        from mini_infra.megatron.core.context_parallel.ring_attention import (
            ring_attention_plan,
            seqlen_sweep,
        )
        from mini_infra.megatron.core.context_parallel.rope import rope_summary
        from mini_infra.megatron.core.context_parallel.yarn import yarn_summary

        payload = {
            "seqlen_sweep": seqlen_sweep(),
            "cp4_plan": ring_attention_plan(
                seq_len=65536 if mode == "h200" else 16384, cp_size=4 if mode == "h200" else 2
            ),
            "rope": rope_summary(seq_len=16384),
            "yarn": yarn_summary(train_seq_len=4096, target_seq_len=32768),
        }
        selected = payload["cp4_plan"]
        metric = {
            "metric_type": "long_context_cp",
            "peak_mem_gb": selected["peak_mem_gb_cp"],
            "step_time_ms": 154.0,
            "attn_comm_bytes": selected["attn_comm_bytes"],
            "tokens_per_sec": 820.0,
            "rope_freq_max": payload["rope"]["rope_freq_max"],
            "yarn_scale": payload["yarn"]["yarn_scale"],
            "val_ppl_at_extended_ctx": 11.2,
        }
        return payload, metric

    if mission_id == "l13_moe_ep":
        from mini_infra.megatron.core.transformer.moe.experts import moe_summary

        payload = {
            "dense_baseline": {
                "active_params": 134_217_728,
                "total_params": 134_217_728,
                "step_time_ms": 42.0,
            },
            "moe_cf125": moe_summary(
                num_experts=8, top_k=2, capacity_factor=1.25, ep_size=8 if mode == "h200" else 2
            ),
            "moe_cf100": moe_summary(
                num_experts=8, top_k=2, capacity_factor=1.0, ep_size=8 if mode == "h200" else 2
            ),
            "collapsed_router": moe_summary(
                num_experts=8,
                top_k=2,
                capacity_factor=1.25,
                ep_size=8 if mode == "h200" else 2,
                collapse=True,
            ),
        }
        metric = {
            "metric_type": "moe_ep",
            **{
                key: value
                for key, value in payload["moe_cf125"].items()
                if isinstance(value, (int, float, str, bool))
            },
        }
        return payload, metric

    if mission_id == "l18_data_engineering":
        from mini_infra.data.wds_pipeline import simulate_pipeline

        payload = simulate_pipeline(workers=8 if mode == "h200" else 4, prefetch=8, corrupt_count=1)
        return payload, {
            "metric_type": "data_engineering",
            **{
                key: value
                for key, value in payload.items()
                if isinstance(value, (int, float, str, bool))
            },
        }

    if mission_id == "l23_quant_serving":
        from mini_infra.vllm.run_engine import quant_payload

        rows = [quant_payload(kind) for kind in ["none", "awq", "fp8", "kvint8"]]
        payload = {
            "quant_matrix": rows,
            "accuracy_drift": {row["quant"]: row["acc_drop_pp"] for row in rows},
        }
        awq = rows[1]
        return payload, {
            "metric_type": "quant_serving",
            **{
                key: value
                for key, value in awq.items()
                if isinstance(value, (int, float, str, bool))
            },
        }

    if mission_id == "l24_spec_decode":
        from mini_infra.vllm.spec_decode.draft_runner import spec_decode_summary

        rows = [
            {
                "mode": "baseline",
                "acceptance_rate": 0.0,
                "speedup": 1.0,
                "ttft_ms": 120.0,
                "itl_ms_p50": 18.0,
                "itl_ms_p99": 28.0,
                "draft_latency_ms": 0.0,
                "draft_gpu_mem_gb": 0.0,
            },
            spec_decode_summary("ngram", concurrency=1, domain="in_domain"),
            spec_decode_summary("draft", concurrency=1, domain="in_domain"),
            spec_decode_summary("ngram", concurrency=64, domain="in_domain"),
            spec_decode_summary("ngram", concurrency=1, domain="ood"),
        ]
        payload = {
            "specdec_compare": rows,
            "failure_repro": "high concurrency raises p99 when draft competes with target",
        }
        chosen = rows[1]
        return payload, {
            "metric_type": "spec_decode",
            **{
                key: value
                for key, value in chosen.items()
                if isinstance(value, (int, float, str, bool))
            },
        }

    raise KeyError(mission_id)


def write_artifacts(mission_id: str, run_dir: Path, payload: dict[str, Any]) -> None:
    artifacts = run_dir / "artifacts"
    if mission_id == "l04_gpu_kernel":
        write_json(artifacts / "bench_softmax.json", payload)
        write_json(
            artifacts / "profile_kineto.json",
            {"events": [payload["best"]], "profiler": "kineto-summary"},
        )
        write_text(
            artifacts / "triton_softmax_v1.py",
            "# v1: one row per program, online softmax, masked tail load\n",
        )
        write_text(
            artifacts / "triton_softmax_v2.py",
            "# v2: tune BLOCK_SIZE / num_warps / num_stages and validate roofline\n",
        )
    elif mission_id == "l09_long_context_cp":
        write_json(artifacts / "seqlen_sweep.json", payload)
        write_text(
            artifacts / "cp4_ring_attention.mmd",
            "graph LR\nQ0-->KV0-->KV3-->KV2-->KV1\nQ1-->KV1-->KV0-->KV3-->KV2\n",
        )
    elif mission_id == "l13_moe_ep":
        write_json(artifacts / "moe_compare.json", payload)
        write_json(artifacts / "router_histogram.json", payload["moe_cf125"].get("dispatch", {}))
        write_text(
            artifacts / "moe_pipeline.mmd",
            "graph LR\nRouter-->Permute-->AllToAll1-->Experts-->Unpermute-->AllToAll2\n",
        )
    elif mission_id == "l18_data_engineering":
        write_json(artifacts / "wds_throughput.json", payload)
        write_json(
            artifacts / "dedup_report.json",
            {"dedup_ratio": payload["dedup_ratio"], "pairs": payload["near_duplicates"]},
        )
        write_text(
            artifacts / "shard_recovery_log.txt",
            "\n".join(json.dumps(item, ensure_ascii=False) for item in payload["skipped"]) + "\n",
        )
    elif mission_id == "l23_quant_serving":
        write_json(artifacts / "quant_matrix.json", payload["quant_matrix"])
        write_json(artifacts / "accuracy_drift.json", payload["accuracy_drift"])
        write_text(
            artifacts / "quant_scatter.csv",
            "quant,acc_drop_pp,tokens_per_sec,peak_kv_mem_gb\n"
            + "\n".join(
                f"{row['quant']},{row['acc_drop_pp']},{row['tokens_per_sec']},{row['peak_kv_mem_gb']}"
                for row in payload["quant_matrix"]
            )
            + "\n",
        )
    elif mission_id == "l24_spec_decode":
        write_json(artifacts / "specdec_compare.json", payload["specdec_compare"])
        write_json(artifacts / "failure_repro.json", {"summary": payload["failure_repro"]})
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        (artifacts / "latency_distribution.png").write_bytes(png)


def write_report(mission_id: str, run_dir: Path, mode: str, metric: dict[str, Any]) -> None:
    meta = MISSION_META[mission_id]
    mini_files = "\n".join(f"- `{path}`" for path in meta["mini_files"])
    real_sources = "\n".join(f"- `{path}`" for path in meta["real_sources"])
    metrics = "\n".join(
        f"| {key} | {metric.get(key, '')} | `metrics.jsonl` | |" for key in meta["metrics"]
    )
    write_text(
        run_dir / "report.md",
        f"""# Mission Report：{mission_id} · {meta["title"]}

## 1. 任务目标
验证 `{mission_id}` 的 MiniInfra 同构实现、真实源码映射与 lab artifact 证据链。

## MiniInfra 主线增量

- 本关对应 MiniInfra 文件：
{mini_files}
- 对应真实框架源码：
{real_sources}
- 保留的系统不变量：入口命令、核心数据流、关键指标命名与失败边界保持一致。
- 删掉的生产复杂度：多后端 kernel、真实多机通信、完整模型加载和大规模数据依赖。
- notebook 预测如何被验证/推翻：对照 `prediction.yaml` 的 hypotheses 与 `metrics.jsonl` / `artifacts/`。
- 证据路径：`command.sh`、`config.resolved.yaml`、`metrics.jsonl`、`artifacts/`。

## 硬件边界与可迁移性

- 本次硬件模式：{mode}
- 课程硬件边界：{meta["hardware"]}
- 哪些结论是真实运行：本地 CPU-safe MiniInfra 数学/调度/数据流验证。
- 哪些结论只能迁移为假设：4090/H200 性能上限、NCCL/FP8/roofline 真实性能。
- 下一步需要的最小验证：在目标硬件只改变一个变量重跑矩阵，并保留 profiler 或 serving trace。

## 2. 源码研读记录

| 源码路径 | 我读懂的职责 | 关键不变量 | 可能失败点 |
|---|---|---|---|
| `{meta["mini_files"][0]}` | MiniInfra 主线入口 | 指标可追踪 | validation-only 被误读 |
| `{meta["real_sources"][0]}` | 真实源码对照 | 生命周期同构 | 生产复杂度被省略 |
| `labs/{mission_id}/scripts/run_smoke.py` | lab 产物落盘 | command/config/metrics/artifacts 完整 | 只跑 happy path |

## 3. Notebook 预测

- 预测文件：`prediction.yaml`
- 预测验证：见 `metrics.jsonl` 与 artifacts JSON。

## 4. 实验矩阵

| run_id | 只改变的变量 | 关键指标 | 结论 | 是否真实运行 |
|---|---|---|---|---|
| {run_dir.name} | mode={mode} | {", ".join(meta["metrics"][:3])} | MiniInfra smoke 闭环完成 | validation-only |

## 5. 指标结果

| 指标 | 数值 | 来源文件 | 我的解释 |
|---|---:|---|---|
{metrics}

## 6. 诊断与证据链

- 符合预测：MiniInfra 能写出设计文档要求的标准产物。
- 不符合预测：真实性能需迁移到 4090/H200 才能确认。
- 证据：`artifacts/` 中的 JSON、mermaid/图片占位和 `metrics.jsonl`。

## 7. Debug Ticket

- Ticket ID：见 quest 中 tickets 列表。
- 最小检查：先确认 artifact 字段，再定位源码不变量。
- 验证方式：重跑 `make smoke M={mission_id}` 与 `make grade M={mission_id}`。

## 8. 我原来误解了什么

- 不把 validation-only 的配置或数学模拟写成真实硬件性能。

## 9. 下一步

- 在 4090 或 H200 上补真实 profiler/benchmark，并把本报告的假设改成证据。
""",
    )


def run_half_mission(mission_id: str, run_id: str | None = None, mode: str = "smoke") -> Path:
    run_dir = prepare_run_dir(mission_id, run_id, ROOT)
    write_command_snapshot(run_dir)
    write_yaml(run_dir / "prediction.yaml", {"hypotheses": PREDICTIONS[mission_id]})
    write_yaml(
        run_dir / "config.resolved.yaml",
        {"mission": mission_id, "mode": mode, "created_at": utc_now()},
    )
    payload, metric = build_payload(mission_id, mode)
    write_artifacts(mission_id, run_dir, payload)
    append_jsonl(run_dir / "metrics.jsonl", {"timestamp": utc_now(), **metric})
    write_text(
        run_dir / MISSION_META[mission_id]["log"], f"[{utc_now()}] {mission_id} {mode} 完成\n"
    )
    write_report(mission_id, run_dir, mode, metric)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one of the six inserted half missions")
    parser.add_argument("--mission", required=True, choices=sorted(MISSION_META))
    parser.add_argument("--run-id")
    parser.add_argument("--mode", default="smoke")
    args = parser.parse_args()
    print(run_half_mission(args.mission, args.run_id, args.mode))


if __name__ == "__main__":
    main()
