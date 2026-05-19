# MiniInfraLM：真实 LLM Infra 项目的最小同构主线

MiniInfraLM 的定位不是“另起一套课程替代系统”，而是从 `github_repo/` 中的真实项目抽取最小可运行骨架。它保持真实项目的目录层次、核心类名、函数名、生命周期和系统边界，只把模型规模、数据量和 GPU runtime 缩小到本地可以快速阅读和验证。

> 核心原则：小而同构。每个 MiniInfra 文件都必须能回答：它在 Megatron、vLLM、SGLang 或 SLiME 中对应哪个真实源码文件，保留了哪个系统不变量，删掉了哪些生产复杂度。

## 主线形态

```text
Megatron-shaped tiny pretrain
  → distributed checkpoint / TP / PP / distributed optimizer metadata
  → vLLM-shaped OpenAI entrypoint / LLMEngine / Scheduler / KVCacheManager
  → SGLang-shaped Scheduler / RadixCache / prefill-decode disaggregation
  → SLiME-shaped train loop / RolloutManager / TrainRayActor / weight sync
  → evidence index / report / capstone handoff
```

这条主线解决 lab 分散的问题：学习者不再每关只跑一个孤立脚本，而是在同一个 mini infra 项目上持续加能力，并始终回到真实框架源码验证。

## 每关默认流程

每个 lab 都必须按同一条证据链推进：

```text
读本关 MiniInfra 同构文件
→ 运行 make mini-infra M=<mission_id> RUN_ID=<run_id>
→ 用 notebook 建立预测
→ 阅读 github_repo/ 对应真实源码
→ 运行 lab smoke/真实框架命令
→ 在报告中写 MiniInfra → 真实源码 → artifact/metrics 三方对照
```

`mini_infra/{distributed,training,serving,rl,planner}` 中的旧 toy/辅助模块只用于建立直觉或做 validation-only 检查；主线源码阅读优先以 `mini_infra/{megatron,vllm,sglang,slime}` 的真实同构目录为准。

## 同构对象

| 真实项目 | MiniInfra 同构目录 | 保留的核心生命周期 |
|---|---|---|
| Megatron-LM | `mini_infra/megatron/` | `pretrain_gpt.py` → `training.pretrain/train_step/training_log` → TP/PP schedule → distributed optimizer → distributed checkpoint |
| vLLM | `mini_infra/vllm/` | OpenAI-compatible entrypoint → `LLMEngine.add_request/step` → Scheduler waiting/running/finished → `KVCacheManager.allocate_slots/free` |
| SGLang | `mini_infra/sglang/` | `Scheduler.run_batch` → `RadixCache.match_prefix/insert/cache_finished_req` → PD `DisaggregationService.route_request/transfer_kv` |
| SLiME | `mini_infra/slime/` | `train(args)` → `RolloutManager.generate` → `TrainRayActor.train` → `TrainRayActor.update_weights` → rollout weight sync |

机器可读对齐表见 `mini_infra/source_alignment.yaml`。课程网页和源码阅读入口可以直接打开这些 `mini_infra/` 文件，并与 `github_repo/` 中的真实文件并排阅读。

## 快速开始

```bash
# 跑某一关的 MiniInfra 主线桥接
make mini-infra M=l19_vllm_serving_baseline RUN_ID=demo

# 跑真实项目最小同构栈：Megatron + vLLM + SGLang + SLiME
make mini-infra-real-stack RUN_ID=demo

# 跑全课 MiniInfra smoke：包含证据链、辅助训练/Serving/RL 入口和真实同构栈
make mini-infra-smoke RUN_ID=demo
```

单独观察某个真实框架生命周期：

```bash
python -m mini_infra.megatron.run_pretrain --run-id demo --tp 2 --pp 2 --distributed-optimizer
python -m mini_infra.vllm.run_engine
python -m mini_infra.sglang.run_scheduler --pd
python -m mini_infra.slime.train --steps 2
```

## 真实源码对照

| MiniInfra 文件 | 真实源码文件 | 阅读问题 |
|---|---|---|
| `mini_infra/megatron/pretrain_gpt.py` | `github_repo/Megatron-LM/pretrain_gpt.py` | model provider、batch provider、loss 与训练框架如何分层？ |
| `mini_infra/megatron/training/training.py` | `github_repo/Megatron-LM/megatron/training/training.py` | `pretrain`、`train_step`、日志和 checkpoint 如何串成训练生命周期？ |
| `mini_infra/megatron/training/checkpointing.py` | `github_repo/Megatron-LM/megatron/training/checkpointing.py` | iteration、模型状态、优化器状态、并行切分如何影响 resume？ |
| `mini_infra/megatron/core/tensor_parallel/layers.py` | `github_repo/Megatron-LM/megatron/core/tensor_parallel/layers.py` | Column/Row parallel 分别切哪一维，何时需要 gather/reduce？ |
| `mini_infra/megatron/core/pipeline_parallel/schedules.py` | `github_repo/Megatron-LM/megatron/core/pipeline_parallel/schedules.py` | microbatch、stage 和 bubble 如何决定流水效率？ |
| `mini_infra/megatron/core/optimizer/distrib_optimizer.py` | `github_repo/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py` | distributed optimizer 切的是参数、梯度还是 optimizer state？ |
| `mini_infra/vllm/entrypoints/openai/api_server.py` | `github_repo/vllm/vllm/entrypoints/openai/api_server.py` | HTTP/OpenAI 请求在哪一层转成 engine request？ |
| `mini_infra/vllm/v1/engine/llm_engine.py` | `github_repo/vllm/vllm/v1/engine/llm_engine.py` | `add_request`、`step`、`RequestOutput` 的边界在哪里？ |
| `mini_infra/vllm/v1/core/sched/scheduler.py` | `github_repo/vllm/vllm/v1/core/sched/scheduler.py` | waiting/running/finished 如何影响 TTFT、ITL 和吞吐？ |
| `mini_infra/vllm/v1/core/kv_cache_manager.py` | `github_repo/vllm/vllm/v1/core/kv_cache_manager.py` | KV block 分配、释放和内存压力如何反馈给 scheduler？ |
| `mini_infra/sglang/srt/mem_cache/radix_cache.py` | `github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py` | prefix namespace、match/insert 和 cache_finished_req 如何决定 hit/miss？ |
| `mini_infra/sglang/srt/managers/scheduler.py` | `github_repo/sglang/python/sglang/srt/managers/scheduler.py` | prefill、decode、cache 命中和 batch 调度如何互相影响？ |
| `mini_infra/sglang/srt/managers/disagg_service.py` | `github_repo/sglang/python/sglang/srt/managers/disagg_service.py` | route request、prefill/decode worker load 和 KV transfer 边界在哪里？ |
| `mini_infra/slime/train.py` | `github_repo/slime/train.py` | rollout manager、actor model、weight sync 在同步训练中谁阻塞谁？ |
| `mini_infra/slime/ray/rollout.py` | `github_repo/slime/slime/ray/rollout.py` | RolloutServer/Manager 如何包住 SGLang engine/router？ |
| `mini_infra/slime/ray/train_actor.py` | `github_repo/slime/slime/ray/train_actor.py` | actor train、save_model、update_weights 与 rollout freshness 如何耦合？ |

## Lab 增量路线

| 阶段 | 对应 lab | MiniInfra 主线增量 | 必须联动的学习材料 |
|---|---|---|---|
| Evidence | L00 | `observability/evidence.py` 统一 command/config/metrics/report 证据链 | 网页 docs card、autograder、自学指南 |
| Local training | L01/L03 | `training/trainer.py` 建立训练循环直觉，再映射到 `megatron/training/training.py` | `n01`、`n02`、`n06`、TorchTitan 源码 |
| Distributed core | L01.5/L02 | `distributed/collectives.py` 建 rank/collective 语义，主线落到 Megatron TP/PP 文件 | `n03`、`n04`、`n05`、Megatron TP/PP 源码 |
| Megatron pretrain | L04/L04.8/L05/L05.8/L06 | `megatron/pretrain_gpt.py`、indexed dataset、TP/PP、parallel_state、distributed optimizer、checkpoint | Megatron pretrain/data/checkpoint 源码、显存与 scaling notebook、AI 框架理解口试 |
| vLLM serving | L07/L07.5 | `vllm/entrypoints/openai/api_server.py`、`LLMEngine`、Scheduler、KVCacheManager | `n07`、`n08`、vLLM benchmark/source map |
| SGLang serving | L08/L09/L09.5 | `sglang/srt/mem_cache/radix_cache.py`、Scheduler、PD service | `n08`、`n09`、SGLang scheduler/PD/metrics 源码 |
| RL bridge | L10/L10.5 | reward parser 接入 rollout response schema，为 SLiME rollout 做前置验证 | `n10`、verl reward wrapper、rollout-only smoke |
| SLiME RL | L11/L11.5 | `slime/train.py`、RolloutManager、TrainRayActor、weight sync、rollout freshness | `n10`、SLiME rollout/train_actor/sglang-config 源码 |
| Delivery | L12 | `reports/build_delivery.py` 聚合 MiniInfra、lab run、debug ticket 和风险登记 | 网页 evidence、final artifacts、能力矩阵 |

## 与网页、Notebook、源码研读的联动

- **网页任务控制台**：`docs/mini_infra.md` 作为 docs card 展示主线；`app/backend/repository.py` 允许读取 `mini_infra/` 源码，方便在 Web UI 中打开最小同构文件。
- **Notebook**：notebook 负责建立可预测模型，例如显存、collective、TP/PP、KV cache、prefill/decode、reward/KL；报告必须把预测回填到对应 MiniInfra 文件和真实源码。
- **源码研读地图**：先读 MiniInfra 同构骨架，再读 `github_repo/` 真实实现；对照关注入口、状态对象、不变量、错误边界和被 MiniInfra 删掉的生产复杂度。
- **Lab README**：每关的 `## MiniInfra 主线增量` 要说明本关修改/观察哪个 MiniInfra 文件，以及真实框架对应路径。
- **AI 框架理解口试**：`docs/AI框架理解评估指南.md` 和 `prompts/framework_understanding_tutor.md` 用来检查 patch 是否能回到 MiniInfra/真实源码/debug ticket 三方关系。
- **Autograder/self-check**：只检查证据是否完整；是否真正理解同构关系，需要在 AI 口试、`report.md` 调用链、实验矩阵和迁移判断中证明。

## 学习者应该学到什么

完成 MiniInfra 主线后，学习者应该能：

1. 说清一个训练/Serving/RL 系统从入口命令到核心状态对象再到 metrics/artifacts 的调用链。
2. 把 Megatron 的 pretrain、TP/PP、distributed optimizer、distributed checkpoint 还原成最小生命周期。
3. 解释 vLLM 请求如何从 OpenAI-compatible facade 进入 engine/scheduler/KV cache，并知道 TTFT/ITL 应该在哪些边界观测。
4. 解释 SGLang prefix cache 与 PD 分离为什么会改变首 token 延迟、decode 延迟和 KV transfer 风险。
5. 解释 SLiME 中 rollout、actor train、weight sync、SGLang engine 之间的阻塞和新鲜度问题。
6. 写出 MiniInfra 与真实框架的差异：哪些结论是系统语义，哪些结论不能外推为真实性能。

## 报告要求

每个 lab 的报告都必须包含这段：

```markdown
## MiniInfra 主线增量

- 本关对应 MiniInfra 文件：
- 对应真实框架源码：
- MiniInfra 同构最小实现保留的系统不变量：
- MiniInfra 删掉或缩小的生产复杂度：
- notebook 预测如何被验证/推翻：
- 证据路径：
```

如果本关只跑了 validation-only 路径，报告必须写清它只证明证据链、配置或系统语义；不能把它当作 Megatron、vLLM、SGLang、SLiME 的真实性能结论。

## Bronze / Silver / Gold

- **Bronze**：能运行对应 MiniInfra 同构文件，生成 command/config/metrics/report，并说出保留的生命周期。
- **Silver**：能把 MiniInfra 同构骨架与真实源码逐段对照，完成一个 debug ticket 或源码改造任务。
- **Gold**：能把同一能力迁移到真实框架或 8×H200，控制 workload、batch、seq_len、并行度和硬件 confounder。

## 推荐学习顺序

1. 跑 `make mini-infra-real-stack RUN_ID=first_pass`，确认主线闭环。
2. 读 `mini_infra/source_alignment.yaml`，找到当前 lab 对应的 MiniInfra 文件和真实源码。
3. 跑对应 notebook，先写运行前预测，再记录实验后偏差。
4. 读 MiniInfra 同构文件，再读 `docs/源码研读地图.md` 中的真实框架源码。
5. 跑 lab smoke 或真实框架命令，保存 `runs/<mission>/<run_id>/`。
6. 在报告中写清 MiniInfra、真实源码、实验指标三者的连接和边界。

## 新增半关同构对象

| 真实对象 | MiniInfra 镜像 | 学习重点 |
|---|---|---|
| Triton tutorials / Megatron fused softmax | `mini_infra/gpu/` | softmax/RMSNorm 内核生命周期 |
| Megatron Core context parallel | `mini_infra/megatron/core/context_parallel/` | RoPE/YaRN + ring attention |
| Megatron Core MoE | `mini_infra/megatron/core/transformer/moe/` | router → permute → all-to-all → experts → unpermute → all-to-all |
| webdataset / datasketch | `mini_infra/data/` | shard/worker 切分、minhash 去重、shard 续读 |
| vLLM / SGLang quantization | `mini_infra/vllm/quant/`, `mini_infra/sglang/quant/` | scale/zero、KV 量化、calibration |
| vLLM speculative | `mini_infra/vllm/spec_decode/` | draft/n-gram → verify → KV 回滚 → acceptance |

## v2 增量同构对象（系统级真实痛点）

这一组同构骨架不属于单一真实框架，而是把 RL Infra 在生产中真实摔过跟头的横切机制做了 CPU 模拟版本，让学生在没卡的环境也能完整跑通对策。对标 [Awesome-ML-SYS-Tutorial](../github_repo/Awesome-ML-SYS-Tutorial/) 系列博客。

| 真实对象 | MiniInfra 镜像 | 学习重点 | 对应 lab |
|---|---|---|---|
| `torch.cuda.memory._record_memory_history` + leak analyzer | `mini_infra/torch/memory_snapshot.py` | alloc/free tracker、按 top-of-stack frame 聚合归因、closure capture 泄露检测 | L02.5 (l02.5) |
| HF tokenizer chat template + verl PR #1668 fixed-base 算法 | `mini_infra/data/multiturn_tokenizer.py` + `_mock_tokenizer.py` | BASE_CONVERSATION 增量 tokenize、loss_mask、think token 边界 | L09.7 (l28.5) |
| slime mismatch K3 KL + TIS/MIS/Geometric IS | `mini_infra/rl/mismatch.py` | K3 KL 估计、ratio clip/truncate、token mask + sequence veto、batch normalize | L10.3 (l29.5) |
| `torch.cuda.graph` + `torch_memory_saver` | `mini_infra/torch/cuda_graph_cache.py` | GraphCache capture/replay + static buffer 复用、MemorySavor pause/resume 释放物理 bytes | L10.7 (l30.5) |
| verl `update_weights_from_tensor` (CUDA IPC handle tuple) | `mini_infra/slime/ipc_weight_sync.py` | handle 序列化（不含数据）、`IPCStoragePool` 模拟共享 storage、gather 不对称、flush_cache 时序 | L11.3 (l32.5) |
| slime batch-GAE chunked parallel | `mini_infra/rl/gae_chunk.py` | naive 反向递推、chunked 并行 + boundary correction 标量传递、与 naive 数值精确等价 | L11.7 (l34.5) |

**关键设计原则**：CPU 模拟版本**保证结构性不变量与生产 GPU 版本一致**——例如 handle 必须 < 1KB（证明不含数据）、graph buffer `data_ptr()` 在多次 replay 间不变、savor pause 后 `physical_bytes() == 0`、按 top frame 聚合在 50 step 漏 free 实验里能定位到唯一真凶。这些性质在 GPU 上完全成立，在 CPU 模拟版上也必须成立。**唯一不能验证的是性能数字**（CUDA Graph 加速比、IPC 吞吐、chunked GAE 100-300× 等），那些必须放到真实硬件验证。

详见每个 v2 lab 的 `patch/task.md` 与对应 notebook（`n19`-`n23`）。整体阅读顺序与简化对照见 [docs/qwen3_omni_reading_guide.md](./qwen3_omni_reading_guide.md)。
