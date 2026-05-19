# Infra Quest · 多模态大模型 Infra 工程训练课程

> 这门课的核心：**每关写一个能跑过单元测试的真补丁，并能把这个 patch 放回真实框架源码主线**。pytest 负责验证代码不变量；AI 框架理解口试负责验证你不是只会补局部 TODO。

## 一句话定位

学完这门课，你的简历可以这样写：

> 我读过 Megatron / SGLang / verl / SLiME 的源码，**给每个都写过一个能跑过单元测试的 patch**，最终用这套技术栈把 Qwen2.5-0.5B 改造成了能同时吃图像和语音的多模态 omni 模型，并以 OpenAI 兼容 endpoint 部署。
>
> **而且我懂"基础设施会以哪些方式骗你"**：K3 KL + TIS/MIS 防训推不一致、CUDA IPC handle tuple 做零拷贝权重同步、CUDA Graph + Memory Savor 给 RL co-locate 让显存、fixed-base chat template 防多轮 loss mask 错位、按 stack 聚合定位 VLM RL OOM 源——这套真实生产事故的对策，是我自己写过 patch 跑过测试拿下来的。

## Patch Track：每关只做一件事

每关的核心动作是 **写一个补丁**：

```bash
# 1. 读任务（每关 task.md 是 patch 契约入口）
cat labs/l05_distributed_primitives/patch/task.md

# 2. 改 starter 代码
$EDITOR labs/l05_distributed_primitives/patch/starter/tp_linear.py

# 3. 跑测试，PASS 即过关（无报告，无 rubric）
make patch-test M=l05_distributed_primitives
```

pytest 全绿代表代码契约通过；框架理解还要用 `docs/AI框架理解评估指南.md` 做源码主线口试。课程不恢复繁重 rubric，但不再把局部 patch 当成完整框架理解。

## 课程结构

**30 个常规 lab + 1 个多模态 Capstone**，按 6 章组织。带 ✦/✦✦ 的为 v2 增量 lab，对标 [Awesome-ML-SYS-Tutorial](github_repo/Awesome-ML-SYS-Tutorial/)，直击生产中真正会摔跟头的工程痛点。

### 第 1 章：环境与 PyTorch 基础

| 关卡 | Patch 内容 | 改哪里 |
|---|---|---|
| L01 PyTorch 系统 | `peak_memory()` 上下文管理器 | `mini_infra/gpu/memory_probe.py` |
| L01.5 NCCL/DDP | 手写 bucketed grad all-reduce hook | `mini_infra/distributed/manual_ddp.py` |
| L01.7 GPU kernel | Online softmax + 融合 dropout (Triton) | `mini_infra/gpu/triton_kernels/softmax.py` |
| L02 分布式原语 | 手写 ColumnParallelLinear / RowParallelLinear | `mini_infra/distributed/tp_linear.py` |
| **L02.5 Memory Snapshot ✦** | 按 stack 聚合定位 VLM RL 显存泄露 | `mini_infra/torch/memory_snapshot.py` |

### 第 2 章：Megatron 预训练主线

| 关卡 | Patch 内容 | 改哪里 |
|---|---|---|
| L03 TorchTitan | Selective activation checkpoint policy | `github_repo/torchtitan/.../parallelize_llama.py` |
| L04 Megatron 预训练 | 新 LR scheduler `cosine_with_restarts` | `github_repo/Megatron-LM/.../optimizer_param_scheduler.py` |
| L04.5 长上下文 CP | Ring attention forward | `mini_infra/distributed/ring_attention.py` |
| L04.8 Megatron 主线 | Megatron-shaped train step | `mini_infra/megatron/training/training.py` |
| L05 Megatron 扩展 | Gradient bucket overlap with backward | `github_repo/Megatron-LM/.../distributed_data_parallel.py` |
| L05.5 MoE/EP | Top-2 router + capacity factor + aux loss | `mini_infra/megatron/moe_router.py` |
| L05.8 Megatron CKPT | Distributed checkpoint save/load | `mini_infra/megatron/training/checkpointing.py` |

### 第 3 章：多模态数据

| 关卡 | Patch 内容 | 改哪里 |
|---|---|---|
| **L06 多模态数据 ★** | 变长 image+audio+text collator | `mini_infra/data/multimodal_collate.py` |
| L06.3 数据工程 | MinHash + LSH 去重 | `mini_infra/data/dedup_minhash.py` |

### 第 4 章：Serving（vLLM / SGLang）

| 关卡 | Patch 内容 | 改哪里 |
|---|---|---|
| L07 vLLM | typical_p sampling | `github_repo/vllm/.../sampler.py` |
| L07.5 vLLM 主线 | Scheduler + KV block manager mini model | `mini_infra/vllm/v1/core/sched/scheduler.py` |
| L08 SGLang | RadixTree 前缀缓存 (insert/match/evict) | `mini_infra/sglang/radix_cache.py` |
| L08.5 量化 | AWQ per-channel scale 校准 | `mini_infra/serving/awq_calibrate.py` |
| L08.7 Spec decode | Draft-then-verify | `mini_infra/serving/spec_decode.py` |
| L09 SGLang PD | 三个 prometheus metric exporter | `github_repo/sglang/.../scheduler.py` |
| L09.5 SGLang 主线 | PD route + KV transfer | `mini_infra/sglang/srt/managers/disagg_service.py` |
| **L09.7 Multi-turn Chat Mask ✦** | Fixed-base delta tokenization (verl PR #1668) | `mini_infra/data/multiturn_tokenizer.py` |

### 第 5 章：RL 与对齐

| 关卡 | Patch 内容 | 改哪里 |
|---|---|---|
| L10 verl RL | Adaptive KL controller (PPO) | `github_repo/verl/.../core_algos.py` |
| **L10.3 Train-Infer Mismatch ✦✦** | K3 KL + TIS + MIS + Geometric IS + Veto | `mini_infra/rl/mismatch.py` |
| L10.5 Rollout smoke | vLLM-shape 异步 rollout 接口 | `mini_infra/rl/rollout.py` |
| **L10.7 CUDA Graph + Savor ✦** | capture/replay + pause/resume | `mini_infra/torch/cuda_graph_cache.py` |
| L11 SLiME | TrainRayActor → Inference 权重同步 | `github_repo/slime/.../weight_sync.py` |
| **L11.3 IPC Weight Sync ✦✦** | handle tuple 共享显存（verl/slime 真实机制） | `mini_infra/slime/ipc_weight_sync.py` |
| L11.5 RL 主线 | Rollout freshness policy | `mini_infra/slime/ray/rollout.py` |
| **L11.7 GAE Chunked Parallel ✦** | 100-300× 加速（数学等价于 naive） | `mini_infra/rl/gae_chunk.py` |

### 第 6 章：Capstone

| 关卡 | Patch 内容 | 改哪里 |
|---|---|---|
| **L12 Capstone ★★** | MM-Tiny-Omni：图像+语音多模态 omni 模型 | 见 §Capstone |

每关 patch 规模 30–150 行 Python，闭环时间 2–4 小时。**有 GPU 更好，无 GPU 也能完成 ~70%**（多数 patch 用 gloo backend / CPU 注册表模拟，跑测试就够）。

`.5/.7/.8` 主线/扩展关卡不是堆代码量，而是把局部 patch 接回真实框架生命周期或补上生产事故对策：Megatron pretrain、parallel state/checkpoint、vLLM scheduler/KV、SGLang cache/PD、RL rollout freshness、训推不一致、CUDA IPC weight sync、CUDA Graph、内存泄露归因、多轮 loss mask、GAE 并行。

## v2 增量速览（✦/✦✦ 标记的新 lab）

6 个新 patch lab + 5 个新 notebook，对标 [Zhaochen Yang 的 Awesome-ML-SYS-Tutorial](github_repo/Awesome-ML-SYS-Tutorial/) 系列博客。

| 关卡 | 配套 notebook | 真实事故来源 |
|---|---|---|
| L02.5 Memory Snapshot | `n21_memory_snapshot_walk` | VLM RL OOM 排查（hook 闭包泄露） |
| L09.7 Multi-turn Chat Mask | `n23_chat_template_multiturn` | verl PR #1668 (Yanbin Jiang) |
| L10.3 Train-Infer Mismatch | `n19_train_infer_mismatch` | slime mismatch blog · Qwen30B-A3B 320 步崩溃 |
| L10.7 CUDA Graph + Savor | `n20_cuda_graph_replay` | slime co-locate offload/upload |
| L11.3 IPC Weight Sync | `n22_weight_sync_handle_tuple` | verl `update_weights_from_tensor` 真实机制 |
| L11.7 GAE Chunked Parallel | — | slime batch-GAE 100-300× 加速 |

✦✦ = RL Infra 工程师必修，✦ = 强烈推荐。三个 Tier-3 进阶扩展（不破坏既有测试）：

- `labs/l20_vllm_scheduler_kv/patch/extension_overlap_scheduler.md` — SGLang `event_loop_overlap` + FutureMap
- `labs/l13_moe_ep/patch/extension_dp_attention.md` — DeepSeek 风格 DP Attention
- `labs/l31_rollout_only_smoke/patch/extension_colocate_vs_disaggregate.md` — RL placement 设计选择

`notebooks/n07_kv_cache.ipynb` 与 `notebooks/n10_rl_kl_reward.ipynb` 也补了"RadixCache vs PagedAttention 是策略层 vs 寻址层"和"K1 vs K3 KL 估计器"两段框架理解。

## Capstone：MM-Tiny-Omni

把 Qwen2.5-0.5B 改造成同时吃 **图像 + 语音 + 文本** 的 omni 模型，并以 OpenAI 兼容 endpoint 部署。

三阶段交付：

- **Stage A**：架构改造 + projector 训练（Megatron 路径）
  - 接 CLIP-ViT-B/32 + Whisper-tiny encoder
  - 训练两个 MLP projector + LLM 最后 4 层
  - **必装** L09.7（多轮 mask）+ L02.5（OOM 调试）
- **Stage B**：推理服务化（SGLang 路径）
  - 多模态 scheduler + RadixCache 复用 image token
  - OpenAI Chat Completions 兼容 endpoint
  - **必装** L10.7（TTFT < 500ms 的 graph + savor）
- **Stage C**：RL 对齐（SLiME 路径）
  - CLIP score reward（图像）+ WER reward（语音）
  - PPO + weight sync 微调
  - **必装** L10.3（防训推崩溃）+ L11.3（高效权重同步）
  - **可选** L11.7（GAE 并行加速）

详见 `labs/l35_multimodal_capstone/README.md` 与 `docs/qwen3_omni_reading_guide.md`（Qwen3-Omni 真实架构 → Capstone 简化版的 trade-off 解读）。

## 快速开始

```bash
# 装依赖
make env ENV=base
make frontend-install

# 起前端 + 后端
make backend     # 一个终端
make app         # 另一个终端

# 看任务列表
make list-missions

# 进入第一关
cat labs/l05_distributed_primitives/patch/task.md
$EDITOR labs/l05_distributed_primitives/patch/starter/tp_linear.py
make patch-test M=l05_distributed_primitives
```

## 卡住怎么办

每关都有三级提示：

```bash
make patch-hint M=<lab>            # 看 TODO 列表 + 关键提示
make patch-show-solution M=<lab>   # 看完整参考解
make patch-test-all                # CI 模式：跑全 31 个 patch lab
IMPL=reference make patch-test-all # 验收所有参考解
```

## 配套资源（可选阅读）

- `notebooks/`：每关对应的概念 notebook（先跑这个再写 patch 容易上手）
- `mini_infra/`：贯穿全课的真实工程最小同构骨架，是补丁靶场也是教材
- `github_repo/`：pinned 真实框架快照（Megatron / vLLM / SGLang / verl / SLiME / TorchTitan / **Awesome-ML-SYS-Tutorial**）
- `tickets/`：debug 题（可选，对死锁/形状错配感兴趣再做）
- `docs/AI框架理解评估指南.md`：给 AI tutor 用的框架理解口试协议；patch-test 之后用它检查源码主线理解
- `docs/qwen3_omni_reading_guide.md` ✦：Qwen3-Omni（Codec/RVQ/Dual AR/Thinker-Talker）→ MM-Tiny-Omni Capstone 的工程简化映射

## 目录结构

```
labs/l*/                          # 30 个 patch lab + L00 环境课 + L12 Capstone
├── README.md                     # 本关入口（指向 patch/）
├── Makefile                      # patch-test / patch-hint / patch-show-solution
├── configs/, scripts/, tickets/  # 配套
└── patch/
    ├── task.md                   # ★ 任务说明（必读）
    ├── starter/                  # ★ 你要改的代码
    ├── reference/                # 参考解
    ├── tests/                    # 自动验证（pytest）
    └── extension_*.md            # 进阶挑战（可选，不破坏 patch-test）

mini_infra/                       # 同构骨架 + 补丁靶场
notebooks/                        # 概念 notebook（含 v2 增量 n19-n23）
github_repo/                      # 真实框架 pinned 快照
docs/                             # 全课文档（含 v2 reading guide）
quests/                           # 任务清单 yaml（前端用）
```

## AI 协作守则

AI 是 copilot，不是 oracle。守则：

```
Observe → Ask → Patch Plan → Human Check → Apply → Test → Explain → Commit
```

禁止让 AI 静默改框架、跳过 pytest、绕过断言。每个 patch 必须自己理解每一行。
