# Notebook 与 Lab 对照

## MiniInfraLM 主线联动

notebook 不只是概念实验；每个 notebook 都要回填到真实项目同构骨架里的对应文件：

| Notebook | MiniInfra 同构文件 | 报告中要连接的指标 |
|---|---|---|
| `n00_env_warmup` | `observability/evidence.py` | command/config/env evidence |
| `n01_gpu_memory_anatomy` | `megatron/training/training.py` | 参数/激活/optimizer 显存估算 |
| `n02_pytorch_profiler` | `training/trainer.py` → `megatron/training/training.py` | forward/backward/optimizer step |
| `n03_ddp_collectives` | `distributed/collectives.py` → `megatron/training/initialize.py` | rank/world_size/all_reduce |
| `n04_tensor_parallel_linear` | `megatron/core/tensor_parallel/layers.py` | Column/Row parallel shard/gather/reduce |
| `n05_pipeline_parallel_bubble` | `megatron/core/pipeline_parallel/schedules.py` | microbatch、stage、bubble ratio |
| `n06_activation_recompute` | `planner/scaling_planner.py` → `megatron/training/training.py` | recompute memory/time trade-off |
| `n07_kv_cache` | `vllm/v1/core/kv_cache_manager.py` | KV block usage 与 max_model_len 风险 |
| `n08_prefill_decode` | `vllm/v1/core/sched/scheduler.py`、`sglang/srt/managers/disagg_service.py` | TTFT/ITL/prefill/decode |
| `n09_prefix_cache` | `sglang/srt/mem_cache/radix_cache.py` | prefix namespace、cache hit/miss |
| `n10_rl_kl_reward` | `rl/reward.py`、`slime/ray/rollout.py`、`slime/ray/train_actor.py` | reward/KL/response length/weight sync |


notebook 的职责是建立系统直觉；最终结论必须回到 lab 的真实命令、metrics 和 artifacts。

## L00 · 环境、Conda、CUDA 与 Git 生存课

- `notebooks/n00_env_warmup.ipynb`：运行前：预测 `LOCAL_RANK` 与 `CUDA_VISIBLE_DEVICES` 映射；运行后：读取 `collect_env.json`，确认环境证据字段；无 GPU 可跑。

## L01 · PyTorch 系统基础与 Profiler

- `notebooks/n01_gpu_memory_anatomy.ipynb`：运行前：先手算参数/梯度/优化器/激活显存；运行后：用 peak_memory_gb 验证估算偏差。
- `notebooks/n02_pytorch_profiler.ipynb`：运行前：理解 CUDA async 与 profiler event；运行后：打开 artifacts/profiler/*.json 标出 dataloader/forward/backward/optimizer。


## L01.5 · NCCL hello 与 2-rank DDP Smoke

- `notebooks/n03_ddp_collectives.ipynb`：运行前：预测 2-rank all-reduce 的求和值；运行后：对照 artifacts/ddp_hello.json 的 backend、world_size、barrier_ok。

## L02 · 分布式原语、DDP、TP 与 PP 沙盘

- `notebooks/n03_ddp_collectives.ipynb`：运行前：先画出 all_reduce/broadcast/barrier 的通信语义；运行后：对照 collectives_demo.py 的每个 rank 日志。
- `notebooks/n04_tensor_parallel_linear.ipynb`：运行前：用矩阵切分理解 Column/Row parallel；运行后：对照 MiniInfra/Megatron `ColumnParallelLinear` 与真实 Megatron layers.py。
- `notebooks/n05_pipeline_parallel_bubble.ipynb`：运行前：计算 bubble fraction；运行后：用 MiniInfra/Megatron schedule 解释 microbatch 变化。

## L03 · TorchTitan 训练框架实验

- `notebooks/n06_activation_recompute.ipynb`：运行前：理解 recompute 的显存/计算 trade-off；运行后：对照 TorchTitan activation_checkpoint 配置，不把它当成免费优化。

## L04 · Megatron 纯文本预训练

- `notebooks/n04_tensor_parallel_linear.ipynb`：运行前：复习 TP 切分矩阵；运行后：解释 h200_tp1.yaml 与 h200_tp4.yaml 的 checkpoint 差异。
- `notebooks/n06_activation_recompute.ipynb`：运行前：理解 recompute 对显存/时间的影响；运行后：在 Megatron 配置中说明是否开启及理由。

## L05 · Megatron 扩展与优化

- `notebooks/n04_tensor_parallel_linear.ipynb`：运行前：复习 TP 通信代价；运行后：解释 TP4 的显存下降与通信增加。
- `notebooks/n05_pipeline_parallel_bubble.ipynb`：运行前：计算 bubble；运行后：为 PP/microbatch 组合给出建议。
- `notebooks/n06_activation_recompute.ipynb`：运行前：理解重算；运行后：把显存收益和 step_time 回归写入表格。

## L06 · Megatron 多模态数据管线

- `notebooks/n01_gpu_memory_anatomy.ipynb`：运行前：复习 batch shape 对显存的影响；运行后：解释多模态 batch 中图像/audio tensor 对显存和吞吐的压力。

## L07 · vLLM 推理服务基线

- `notebooks/n07_kv_cache.ipynb`：运行前：理解 KV cache 显存随 batch/seq 增长；运行后：解释 vLLM memory_pressure ticket。
- `notebooks/n08_prefill_decode.ipynb`：运行前：拆分 prefill 与 decode；运行后：报告中分别写 TTFT 与 ITL。

## L08 · SGLang Serving 核心

- `notebooks/n07_kv_cache.ipynb`：运行前：理解 KV cache 内存模型；运行后：解释 SGLang cache hit/miss 与显存压力。
- `notebooks/n09_prefix_cache.ipynb`：运行前：手算 repeated prefix 命中条件；运行后：用 bench_repeated_prefix.py 验证 cache_hit_rate。

## L09 · SGLang PD 分离与可观测性

- `notebooks/n08_prefill_decode.ipynb`：运行前：拆分 TTFT/ITL；运行后：把 unified、2P6D、4P4D 指标映射到 prefill/decode。
- `notebooks/n09_prefix_cache.ipynb`：运行前：理解 prefix reuse；运行后：解释 PD 下 cache/路由对 TTFT 的影响。

## L10 · verl RL baseline 与 Reward/KL

- `notebooks/n10_rl_kl_reward.ipynb`：运行前：先理解 reward、KL、response length 的相互作用；运行后：用 reward_self_test 和 metrics 判断 RL baseline 是否可信。


## L10.5 · Rollout-only Smoke

- `notebooks/n10_rl_kl_reward.ipynb`：运行前：预测 response 长度、reward parser 与 KL 风险；运行后：把 rollout-only response 样例放进 reward parser，确认 schema 可迁移到 L11。

## L11 · SLiME RL 核心

- `notebooks/n10_rl_kl_reward.ipynb`：运行前：理解 reward/KL/rollout 的基本闭环；运行后：解释 SLiME 中 reward、actor update、weight sync 的系统成本。

## L12 · 多模态 Infra Capstone

- `notebooks/n01_gpu_memory_anatomy.ipynb → n10_rl_kl_reward.ipynb`：运行前：回顾所有 notebook 的系统直觉；运行后：在 final_readme 中把每个直觉映射到真实 artifact。

## 新增半关 Notebooks

| Notebook | 对应关卡 | 用途 |
|---|---|---|
| `n11_gpu_memory_hierarchy.ipynb` | L01.7 | HBM/L2/SMEM/Reg 与 roofline 数量级 |
| `n12_triton_softmax_walkthrough.ipynb` | L01.7 | online softmax、mask 与数值一致性 |
| `n13_rope_yarn_math.ipynb` | L04.5 | RoPE 频率与 YaRN scale |
| `n14_context_parallel_ringattn.ipynb` | L04.5 | CP ring attention 通信与累加路径 |
| `n15_moe_router_capacity.ipynb` | L05.5 | top-k router、capacity、overflow 与 entropy |
| `n16_minhash_dedup.ipynb` | L06.3 | MinHash 近重复与阈值校准 |
| `n17_quant_calibration.ipynb` | L08.5 | SmoothQuant/AWQ calibration 与 scale |
| `n18_spec_decode_acceptance.ipynb` | L08.7 | acceptance rate 与 expected speedup |

## v2 增量 Notebooks（参考 Awesome-ML-SYS-Tutorial）

直击 RL Infra 在生产中真正会摔跟头的 5 个系统级痛点，建议在做对应 lab 之前各跑 30–60 分钟：

| Notebook | 对应关卡 | 用途 | 真实事故来源 |
|---|---|---|---|
| `n19_train_infer_mismatch.ipynb` | L29.5 | K3 KL vs K1，TIS / MIS 直觉，何时干预 | slime mismatch 博客 · Qwen30B-A3B 320 步崩溃 |
| `n20_cuda_graph_replay.ipynb` | L30.5 | capture/replay 直觉，prefill vs decode sweet spot | slime co-locate offload/upload |
| `n21_memory_snapshot_walk.ipynb` | L02.5 | 按 stack 聚合定位泄露，区分真泄露 vs caching | VLM RL 训练 OOM 现场 |
| `n22_weight_sync_handle_tuple.ipynb` | L32.5 | 三种 weight sync 接口对比 + handle tuple 链路 | RL 系统深思博客 · verl/slime/AReaL |
| `n23_chat_template_multiturn.ipynb` | L28.5 | 复现 chat template 的两个噩梦 + Fixed Base 解法 | verl PR #1668（Yanbin Jiang）|

n07/n10 也补了两段框架理解（n07 §3.5 RadixCache vs PagedAttention「策略层 vs 寻址层」；n10 §3.5 K1 vs K3 KL 估计器选择），不需要单独跑。

## v2 增量：系统级真实痛点 Notebooks

这一组 notebook 配合 v2 的 5 个新 lab，用 CPU 模拟实现可视化关键概念。每个都
直接 `import` 对应 lab 的 `reference/` 实现，跑通后再去做 patch 会更顺。

| Notebook | 对应 lab | 用途 | 真实事故 |
|---|---|---|---|
| `n19_train_infer_mismatch.ipynb` | L10.3 (l29.5) | K1 vs K3 KL 方差对比；模拟 rollout 与 training logp 漂移；MIS veto 在 ratio=3e9 时整条丢弃 | slime mismatch · Qwen30B-A3B 320 步崩溃 |
| `n20_cuda_graph_replay.ipynb` | L10.7 (l30.5) | static buffer 复用证据；multi-bs cache 命中；savor pause/resume 物理 bytes 归零并复原 | slime co-locate offload/upload 节奏 |
| `n21_memory_snapshot_walk.ipynb` | L01.3 (l02.5) | 50 步 VLM 训练复现 hook 漏 free，按 top-frame 聚合自动归因到真凶 | VLM RL OOM 排查 |
| `n22_weight_sync_handle_tuple.ipynb` | L11.3 (l32.5) | 1024×1024 fp32 tensor 的 handle 实测 < 200B；多 rank deserialize 共享 storage；slime 分桶显存账 | verl `update_weights_from_tensor` 内部机制 |
| `n23_chat_template_multiturn.ipynb` | L09.7 (l28.5) | 三大暗坑逐个演示（默认 system 注入 / BPE 合并 / think strip），最后用 fixed-base 解决 | verl PR #1668 (Yanbin Jiang) |

补：`n07_kv_cache.ipynb` 增补了 "RadixCache vs PagedAttention：策略层 vs 寻址层"
（Zhaochen Yang 的 framing），`n10_rl_kl_reward.ipynb` 增补了 "K1 vs K3 KL 估计器"
对比表，串到 N19 / L29.5。
