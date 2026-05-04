# MiniInfraLM：真实 LLM Infra 的最小同构项目

MiniInfraLM 是 InfraCourse 贯穿全课的 mini 项目。它不是替代 Megatron、vLLM、SGLang、verl 或 SLiME，也不是另写一套孤立练习系统；它从 `github_repo/` 的真实项目抽取最小可运行骨架，保留目录、类名、函数名、生命周期和系统边界。

## 一条主线

```text
Megatron-shaped pretrain
→ distributed checkpoint / TP / PP / distributed optimizer
→ vLLM-shaped OpenAI entrypoint / LLMEngine / Scheduler / KV cache
→ SGLang-shaped Scheduler / RadixCache / prefill-decode disaggregation
→ SLiME-shaped rollout / TrainRayActor / weight sync
→ delivery evidence bundle
```

## 快速运行

```bash
# 每关默认入口：按 mission 跑对应 MiniInfra 同构桥接
make mini-infra M=l19_vllm_serving_baseline RUN_ID=demo

# 全栈 smoke：Megatron + vLLM + SGLang + SLiME
make mini-infra-real-stack RUN_ID=demo
make mini-infra-smoke RUN_ID=demo
```

单独运行各段：

```bash
python -m mini_infra.megatron.run_pretrain --run-id demo --tp 2 --pp 2 --distributed-optimizer
python -m mini_infra.vllm.run_engine
python -m mini_infra.sglang.run_scheduler --pd
python -m mini_infra.slime.train --steps 2
```

产物会写入：

```text
runs/mini_infra/
├── megatron/<run_id>/
├── real_stack/<run_id>.json
├── train/<run_id>_train/
├── serving/<run_id>_serving/
├── rl/<run_id>_rl/
└── delivery/<run_id>/
```

## 同构核心模块

| MiniInfra 模块 | 对应真实源码 | 对应 lab |
|---|---|---|
| `megatron/pretrain_gpt.py` | `github_repo/Megatron-LM/pretrain_gpt.py` | L04 |
| `megatron/training/training.py` | `github_repo/Megatron-LM/megatron/training/training.py` | L04/L05 |
| `megatron/training/checkpointing.py` | `github_repo/Megatron-LM/megatron/training/checkpointing.py` | L03/L04/L11 |
| `megatron/core/tensor_parallel/layers.py` | `github_repo/Megatron-LM/megatron/core/tensor_parallel/layers.py` | L02/L05 |
| `megatron/core/pipeline_parallel/schedules.py` | `github_repo/Megatron-LM/megatron/core/pipeline_parallel/schedules.py` | L02/L05 |
| `megatron/core/optimizer/distrib_optimizer.py` | `github_repo/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py` | L05 |
| `vllm/entrypoints/openai/api_server.py` | `github_repo/vllm/vllm/entrypoints/openai/api_server.py` | L07 |
| `vllm/v1/engine/llm_engine.py` | `github_repo/vllm/vllm/v1/engine/llm_engine.py` | L07 |
| `vllm/v1/core/sched/scheduler.py` | `github_repo/vllm/vllm/v1/core/sched/scheduler.py` | L07 |
| `vllm/v1/core/kv_cache_manager.py` | `github_repo/vllm/vllm/v1/core/kv_cache_manager.py` | L07/L08 |
| `sglang/srt/mem_cache/radix_cache.py` | `github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py` | L08 |
| `sglang/srt/managers/scheduler.py` | `github_repo/sglang/python/sglang/srt/managers/scheduler.py` | L08 |
| `sglang/srt/managers/disagg_service.py` | `github_repo/sglang/python/sglang/srt/managers/disagg_service.py` | L09 |
| `slime/train.py` | `github_repo/slime/train.py` | L11 |
| `slime/ray/rollout.py` | `github_repo/slime/slime/ray/rollout.py` | L10.5/L11 |
| `slime/ray/train_actor.py` | `github_repo/slime/slime/ray/train_actor.py` | L11 |

完整映射见 `mini_infra/source_alignment.yaml` 和 `docs/mini_infra.md`。

## 辅助模块

`mini_infra/data/`、`training/`、`distributed/`、`serving/`、`rl/`、`planner/`、`reports/` 保留为课程辅助层：用于生成证据、建立 notebook 直觉、做 validation-only 检查和 L12 交付聚合。主线源码阅读应优先从 `mini_infra/{megatron,vllm,sglang,slime}/` 开始。

## 学习边界

- MiniInfra 保留真实项目的系统语义和生命周期，不证明真实性能。
- tiny 数据、tiny 模型和本地 runtime 只用于快速验证调用链、不变量和失败边界。
- 性能结论必须回到对应 lab 的 Megatron、vLLM、SGLang 或 SLiME 真实运行证据。
- 报告必须说明 MiniInfra 保留了什么、删掉了什么、哪些结论可以迁移到真实项目。
