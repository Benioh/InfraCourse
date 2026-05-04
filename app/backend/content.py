from __future__ import annotations

CONCEPT_MAP = {
    "micro_batch_size": {
        "definition": "每张卡每次 forward/backward 处理的样本数。",
        "why_it_matters": "它直接影响激活显存，并与 gradient accumulation 一起决定 global batch。",
        "where_it_appears": ["L01", "L04", "L05", "L11"],
        "related_experiments": ["L01 长序列显存实验", "L04 batch 公式", "L05 扩展估算"],
        "common_failure": "只增大 micro batch 导致 OOM，却误以为是模型权重太大。",
    },
    "global_batch_size": {
        "definition": "一次 optimizer step 实际覆盖的全局样本数。",
        "why_it_matters": "它由 micro batch、data parallel size 和 gradient accumulation 共同决定，影响收敛与吞吐。",
        "where_it_appears": ["L03", "L04", "L05"],
        "related_experiments": ["TorchTitan 配置", "Megatron TP/DP 对照"],
        "common_failure": "并行度变化后没有同步更新 accumulation，导致训练语义漂移。",
    },
    "activation_memory": {
        "definition": "反向传播需要保留或重算的中间激活显存。",
        "why_it_matters": "长序列和大 batch 下，它常常比参数显存更先触顶。",
        "where_it_appears": ["L01", "L05"],
        "related_experiments": ["N01 显存解剖", "N06 activation recompute"],
        "common_failure": "遇到 OOM 先盲目缩模型，而不是检查 seq_len、batch 和 checkpointing。",
    },
    "all_reduce": {
        "definition": "所有 rank 参与规约并拿到相同结果的 collective。",
        "why_it_matters": "DDP 梯度同步依赖 all-reduce；任何 rank 顺序不一致都会 hang。",
        "where_it_appears": ["L02", "L03", "L04"],
        "related_experiments": ["collectives_demo.py", "ddp_toy_train.py"],
        "common_failure": "某个 rank 条件分支少调用 collective，导致其它 rank 永久等待。",
    },
    "tensor_parallel": {
        "definition": "把矩阵乘法或 attention 张量切到多张卡上计算。",
        "why_it_matters": "它降低单卡权重/激活压力，但引入 gather/reduce-scatter 通信。",
        "where_it_appears": ["L02", "L04", "L05"],
        "related_experiments": ["N04 tensor parallel linear", "Megatron TP 配置"],
        "common_failure": "修改 TP 后直接 resume 旧 checkpoint，导致分片形状不兼容。",
    },
    "pipeline_parallel": {
        "definition": "把模型层切到多个 stage，通过 microbatch 流水执行。",
        "why_it_matters": "它能扩展更深模型，但会产生 pipeline bubble 和调度复杂度。",
        "where_it_appears": ["L02", "L05"],
        "related_experiments": ["N05 pipeline bubble"],
        "common_failure": "microbatch 太少导致 bubble 高，却误判为 GPU 算力不足。",
    },
    "checkpoint_compatibility": {
        "definition": "checkpoint 能否在相同模型、tokenizer、并行切分与优化器语义下恢复。",
        "why_it_matters": "保存文件存在不代表可恢复；恢复失败会浪费大规模训练成本。",
        "where_it_appears": ["L03", "L04", "L11", "L12"],
        "related_experiments": ["TorchTitan resume", "Megatron resume"],
        "common_failure": "改 tokenizer、TP 或模型结构后仍尝试加载旧 checkpoint。",
    },
    "indexed_dataset": {
        "definition": "Megatron 预处理后生成的 .bin/.idx 数据格式。",
        "why_it_matters": "训练时 `--data-path` 指向 indexed dataset 前缀，而不是原始 JSONL。",
        "where_it_appears": ["L04"],
        "related_experiments": ["WikiText JSONL", "preprocess_megatron.sh"],
        "common_failure": "路径指到 .jsonl 或缺少 .idx/.bin，训练启动后才报错。",
    },
    "webdataset_shard": {
        "definition": "把多模态样本打包成 tar shard 的数据组织方式。",
        "why_it_matters": "它让顺序读取、分布式切分和样本完整性更可控。",
        "where_it_appears": ["L06"],
        "related_experiments": ["build_webdataset_shards.py", "inspect_shards.py"],
        "common_failure": "同一 sample key 下缺少 image/audio/text 任一文件。",
    },
    "ttft": {
        "definition": "Time To First Token，首 token 延迟。",
        "why_it_matters": "用户体感强相关，长 prompt 下通常由 prefill 主导。",
        "where_it_appears": ["L07", "L08", "L09"],
        "related_experiments": [
            "vLLM baseline",
            "SGLang repeated-prefix",
            "PD prefill",
        ],
        "common_failure": "只看 tokens/s，不看首 token 延迟回归。",
    },
    "itl": {
        "definition": "Inter-Token Latency，decode 阶段相邻 token 延迟。",
        "why_it_matters": "长输出体验由 ITL 决定，常受 KV cache 和 decode 调度影响。",
        "where_it_appears": ["L07", "L08", "L09"],
        "related_experiments": ["long decode benchmark"],
        "common_failure": "prefill 优化后 ITL 变差，却没有分阶段观察。",
    },
    "prefix_cache": {
        "definition": "复用重复 prompt 前缀的 KV 计算结果。",
        "why_it_matters": "相同系统提示或长上下文复用时可显著降低 TTFT。",
        "where_it_appears": ["L08", "L09"],
        "related_experiments": ["N09 prefix cache", "bench_repeated_prefix.py"],
        "common_failure": "空格、标点或 chat template 差异导致看似相同的前缀实际 miss。",
    },
    "pd_disaggregation": {
        "definition": "把 prefill 和 decode 放到不同 engine 或 GPU 池上执行。",
        "why_it_matters": "两阶段资源画像不同，分离后可能降低混合负载排队。",
        "where_it_appears": ["L09"],
        "related_experiments": ["unified vs 2P6D vs 4P4D"],
        "common_failure": "router misroute 或 decode GPU 过少导致 starvation。",
    },
    "reward_parser": {
        "definition": "把模型输出解析为可评分答案的规则或模型。",
        "why_it_matters": "奖励解析错了，RL 目标就错了。",
        "where_it_appears": ["L10", "L11", "L12"],
        "related_experiments": ["GSM8K final number reward"],
        "common_failure": "解析到中间数字而不是最终答案，reward_mean 失真。",
    },
    "kl_penalty": {
        "definition": "约束策略模型不要偏离参考模型过远的正则项。",
        "why_it_matters": "它稳定 RL，但过大或过小都会破坏训练信号。",
        "where_it_appears": ["L10", "L11"],
        "related_experiments": ["N10 reward/KL"],
        "common_failure": "KL 爆炸压过 reward，或者 KL 太低导致长度/风格失控。",
    },
    "weight_sync": {
        "definition": "把 actor 更新后的权重同步到 rollout engine。",
        "why_it_matters": "它决定 rollout 新鲜度，也可能成为 SLiME 吞吐瓶颈。",
        "where_it_appears": ["L11", "L12"],
        "related_experiments": ["actor/rollout split"],
        "common_failure": "sync 太慢导致 rollout 等待，GPU 资源被空耗。",
    },
}

SOURCE_MAPS = {
    "pytorch": [
        {
            "node": "L01 tiny transformer training loop",
            "what_to_search": "DataLoader, forward, backward, optimizer.step, profiler, metrics.jsonl",
            "what_to_inspect": "labs/l02_pytorch_systems/scripts/train_tiny_transformer.py",
            "question": "step_time_ms、tokens_per_sec、peak_memory_gb 分别在哪里计算，哪些计时需要 CUDA synchronize？",
        },
        {
            "node": "L02 torch distributed collectives",
            "what_to_search": "init_process_group, all_reduce, broadcast, barrier, destroy_process_group",
            "what_to_inspect": "labs/l05_distributed_primitives/scripts/collectives_demo.py",
            "question": "哪个 collective 顺序不一致最容易造成 rank hang，日志如何证明所有 rank 到达同步点？",
        },
        {
            "node": "L02 DDP toy train",
            "what_to_search": "DistributedDataParallel, DistributedSampler, rank0-only write, loss aggregation",
            "what_to_inspect": "labs/l05_distributed_primitives/scripts/ddp_toy_train.py",
            "question": "DDP 同步的是梯度还是参数，sampler/seed 如何影响可复现性？",
        },
    ],
    "TorchTitan": [
        {
            "node": "CLI to Trainer boundary",
            "what_to_search": "config, Trainer, seed checkpoint, train_step, train loop",
            "what_to_inspect": "github_repo/torchtitan/torchtitan/train.py; github_repo/torchtitan/torchtitan/trainer.py",
            "question": "TorchTitan 如何把 L01 手写 step 封装成可配置、可 checkpoint、可并行的训练系统？",
        },
        {
            "node": "CheckpointManager",
            "what_to_search": "save, load, async staging, last_step, model-only checkpoint, distributed checkpoint",
            "what_to_inspect": "github_repo/torchtitan/torchtitan/components/checkpoint.py",
            "question": "checkpoint 文件存在为什么不等价于可 resume，model-only 与 full checkpoint 风险分别是什么？",
        },
        {
            "node": "MetricsProcessor",
            "what_to_search": "memory monitor, tokens/sec, step time, logging sink",
            "what_to_inspect": "github_repo/torchtitan/torchtitan/components/metrics.py",
            "question": "哪些指标来自 GPU，哪些来自训练循环，如何和 L01 profiler 证据对齐？",
        },
    ],
    "MiniInfra": [
        {
            "node": "Megatron-shaped pretrain lifecycle",
            "what_to_search": "model_provider, pretrain, train_step, training_log, save_checkpoint, ColumnParallelLinear, DistributedOptimizer",
            "what_to_inspect": "mini_infra/megatron/pretrain_gpt.py; mini_infra/megatron/training/training.py; mini_infra/megatron/core/tensor_parallel/layers.py; mini_infra/megatron/core/optimizer/distrib_optimizer.py; github_repo/Megatron-LM/pretrain_gpt.py",
            "question": "MiniInfra 保留了 Megatron 训练生命周期的哪些不变量，删掉了哪些生产复杂度？",
        },
        {
            "node": "vLLM-shaped request lifecycle",
            "what_to_search": "OpenAIServingChat, LLMEngine.add_request, step, Scheduler, KVCacheManager.allocate_slots",
            "what_to_inspect": "mini_infra/vllm/entrypoints/openai/api_server.py; mini_infra/vllm/v1/engine/llm_engine.py; mini_infra/vllm/v1/core/sched/scheduler.py; mini_infra/vllm/v1/core/kv_cache_manager.py; github_repo/vllm/vllm/v1/engine/llm_engine.py",
            "question": "请求从 OpenAI facade 到 scheduler/KV cache 经过哪些状态，TTFT/ITL 应在哪些边界观察？",
        },
        {
            "node": "SGLang-shaped cache and PD lifecycle",
            "what_to_search": "Scheduler.run_batch, RadixCache.match_prefix, InsertParams, cache_finished_req, route_request, transfer_kv",
            "what_to_inspect": "mini_infra/sglang/srt/managers/scheduler.py; mini_infra/sglang/srt/mem_cache/radix_cache.py; mini_infra/sglang/srt/managers/disagg_service.py; github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py",
            "question": "prefix cache 命中、prefill/decode worker load 和 KV transfer 如何影响 TTFT/ITL？",
        },
        {
            "node": "SLiME-shaped rollout and actor loop",
            "what_to_search": "train, RolloutManager.generate, RolloutServer, TrainRayActor.train, update_weights",
            "what_to_inspect": "mini_infra/slime/train.py; mini_infra/slime/ray/rollout.py; mini_infra/slime/ray/train_actor.py; github_repo/slime/train.py; github_repo/slime/slime/ray/train_actor.py",
            "question": "rollout、actor update、weight sync 谁阻塞谁，哪些状态决定 rollout freshness？",
        },
    ],
    "Megatron": [
        {
            "node": "GPT pretrain entry",
            "what_to_search": "model_provider, get_batch, loss_func, pretrain",
            "what_to_inspect": "github_repo/Megatron-LM/pretrain_gpt.py",
            "question": "数据 iterator、模型构建、loss/token shift 分别由哪层负责？",
        },
        {
            "node": "Megatron training loop",
            "what_to_search": "pretrain, train_step, forward_backward_func, save_checkpoint_and_time, throughput",
            "what_to_inspect": "github_repo/Megatron-LM/megatron/training/training.py",
            "question": "global batch、forward/backward schedule、日志/MFU 和 checkpoint 调用如何连接？",
        },
        {
            "node": "Data preprocessing and indexed dataset",
            "what_to_search": "JSONL text, tokenizer, indexed dataset builder, .bin, .idx, --data-path",
            "what_to_inspect": "github_repo/Megatron-LM/tools/preprocess_data.py; github_repo/Megatron-LM/megatron/core/datasets/indexed_dataset.py",
            "question": "为什么训练时 --data-path 指向 indexed dataset prefix，而不是原始 JSONL？",
        },
        {
            "node": "Tensor and pipeline parallel core",
            "what_to_search": "parallel_state, ColumnParallelLinear, RowParallelLinear, 1F1B, pipeline bubble",
            "what_to_inspect": "github_repo/Megatron-LM/megatron/core/parallel_state.py; github_repo/Megatron-LM/megatron/core/tensor_parallel/layers.py; github_repo/Megatron-LM/megatron/core/pipeline_parallel/schedules.py",
            "question": "TP/PP/DP 如何分解 world size，哪些配置会改变 checkpoint 分片语义？",
        },
        {
            "node": "Multimodal data path",
            "what_to_search": "preprocess_mmdata, WebDataset, Energon, multimodal dataloader, sample key",
            "what_to_inspect": "github_repo/Megatron-LM/tools/preprocess_mmdata.py; github_repo/Megatron-LM/examples/multimodal/dataloader_provider.py",
            "question": "多模态数据如何在训练前验证 schema、shard 完整性和 batch shape？",
        },
    ],
    "vLLM": [
        {
            "node": "OpenAI-compatible server",
            "what_to_search": "api_server, OpenAIServing, engine client, route, request validation",
            "what_to_inspect": "github_repo/vllm/vllm/entrypoints/openai/api_server.py",
            "question": "OpenAI 请求在哪一层转成 engine request，模型加载失败和请求失败如何区分？",
        },
        {
            "node": "v1 engine and scheduler",
            "what_to_search": "LLMEngine.add_request, step, Scheduler, waiting/running, token budget",
            "what_to_inspect": "github_repo/vllm/vllm/v1/engine/llm_engine.py; github_repo/vllm/vllm/v1/core/sched/scheduler.py",
            "question": "请求从 add_request 到 RequestOutput 经过哪些状态，TTFT/ITL 应该在哪些边界观测？",
        },
        {
            "node": "Serving benchmark",
            "what_to_search": "benchmark_serving, concurrency, TTFT, ITL, request latency, throughput",
            "what_to_inspect": "github_repo/vllm/benchmarks/benchmark_serving.py",
            "question": "benchmark 如何避免只测客户端，如何区分 TTFT、ITL 和端到端 latency？",
        },
    ],
    "SGLang": [
        {
            "node": "Server launch and HTTP entrypoint",
            "what_to_search": "launch_server, http_server, OpenAI route, streaming, model path, tensor parallel",
            "what_to_inspect": "github_repo/sglang/python/sglang/launch_server.py; github_repo/sglang/python/sglang/srt/entrypoints/http_server.py",
            "question": "SGLang 启动成功、模型可服务、请求成功分别需要哪些证据？",
        },
        {
            "node": "Scheduler and prefix cache",
            "what_to_search": "Scheduler, waiting/running batch, prefill, decode, RadixCache.match_prefix, cache hit",
            "what_to_inspect": "github_repo/sglang/python/sglang/srt/managers/scheduler.py; github_repo/sglang/python/sglang/srt/mem_cache/radix_cache.py",
            "question": "repeated-prefix workload 如何证明 cache 生效，空格/chat template 为什么会造成 miss？",
        },
        {
            "node": "PD disaggregation and metrics",
            "what_to_search": "pd_disaggregation, disagg_service, router, prefill, decode, metrics_collector",
            "what_to_inspect": "github_repo/sglang/docs/advanced_features/pd_disaggregation.md; github_repo/sglang/python/sglang/srt/managers/disagg_service.py; github_repo/sglang/python/sglang/srt/observability/metrics_collector.py",
            "question": "TTFT 与 ITL 分别受 prefill、decode、router 哪个边界影响最大？",
        },
    ],
    "verl": [
        {
            "node": "GSM8K reward pipeline",
            "what_to_search": "prepare_gsm8k_prompts, reward_math, final answer parser, reward_self_test",
            "what_to_inspect": "labs/l29_verl_rl_baseline/scripts/prepare_gsm8k_prompts.py; labs/l29_verl_rl_baseline/scripts/reward_math.py",
            "question": "reward parser self-test 是否覆盖最终答案提取，错误 reward 如何伪装成 RL 收敛？",
        },
        {
            "node": "verl lab wrapper boundary",
            "what_to_search": "run_verl_lab, validation-only, rl.log, reward_mean, kl_mean, rollout_time",
            "what_to_inspect": "labs/l29_verl_rl_baseline/scripts/run_verl_lab.py",
            "question": "哪些结果是真实 verl 运行，哪些只是本课程 wrapper validation？",
        },
    ],
    "SLiME": [
        {
            "node": "SLiME train loop",
            "what_to_search": "RolloutManager, actor_model.update_weights, generate, train, update interval",
            "what_to_inspect": "github_repo/slime/train.py; github_repo/slime/train_async.py",
            "question": "同步/异步训练里 rollout、actor update、weight sync 谁阻塞谁？",
        },
        {
            "node": "Ray rollout and SGLang engines",
            "what_to_search": "RolloutServer, RolloutManager, sglang router, rollout_num_gpus_per_engine, update_weights",
            "what_to_inspect": "github_repo/slime/slime/ray/rollout.py",
            "question": "改变 rollout GPU 数量会如何影响 engine 数、SGLang TP、rollout throughput 和 freshness？",
        },
        {
            "node": "SGLang rollout client and reward",
            "what_to_search": "generate, generate_and_rm, meta_info, trace attrs, reward model",
            "what_to_inspect": "github_repo/slime/slime/rollout/sglang_rollout.py",
            "question": "SGLang meta_info 中哪些字段应进入 RL metrics，reward collapse 和 rollout bottleneck 如何区分？",
        },
        {
            "node": "SLiME SGLang config",
            "what_to_search": "sglang-config, --sglang-*, engine split, mixed offload, deterministic inference",
            "what_to_inspect": "github_repo/slime/docs/zh/advanced/sglang-config.md",
            "question": "哪些 SGLang 参数转发会导致 server 启动成功但 rollout 语义错误？",
        },
    ],
}

CONCEPT_MAP.update(
    {
        "hbm_bandwidth": {
            "definition": "GPU HBM 全局显存读写带宽，L01.7 用它解释 fused softmax 的主要收益。",
            "why_it_matters": "长序列 softmax 往往 memory-bound，减少 HBM 往返比增加算力更关键。",
            "where_it_appears": ["L01.7"],
            "related_experiments": ["n11_gpu_memory_hierarchy", "bench_softmax.json"],
            "common_failure": "把 validation-only roofline 估算写成真实硬件带宽。",
        },
        "context_parallel": {
            "definition": "沿序列维切分上下文，通过 ring attention 交换 KV 的长上下文并行方式。",
            "why_it_matters": "它降低单 rank attention 显存，但引入随序列增长的通信。",
            "where_it_appears": ["L04.5"],
            "related_experiments": ["n14_context_parallel_ringattn", "seqlen_sweep.json"],
            "common_failure": "mask、position ids 或 seq chunk 未按 CP 维度一致切分。",
        },
        "expert_parallel": {
            "definition": "MoE 中把 expert 分散到不同 rank，通过 all-to-all dispatch token。",
            "why_it_matters": "它扩大总参数但通信和负载均衡会决定吞吐。",
            "where_it_appears": ["L05.5"],
            "related_experiments": ["n15_moe_router_capacity", "moe_compare.json"],
            "common_failure": "router collapse 或 capacity overflow 被误判成模型质量问题。",
        },
        "acceptance_rate": {
            "definition": "speculative decoding 中 draft token 被 target 接受的比例。",
            "why_it_matters": "acceptance 不够高时 draft 成本会抵消加速收益。",
            "where_it_appears": ["L08.7"],
            "related_experiments": ["n18_spec_decode_acceptance", "specdec_compare.json"],
            "common_failure": "只看平均 ITL，不看高并发 p99 和 rollback 成本。",
        },
    }
)

SOURCE_MAPS.setdefault("MiniInfra", []).extend(
    [
        {
            "node": "GPU kernel and roofline",
            "what_to_search": "HBM, SMEM, online softmax, BLOCK_SIZE, occupancy, mask",
            "what_to_inspect": "mini_infra/gpu/memory_model.py; mini_infra/gpu/triton_softmax.py; mini_infra/gpu/microbench.py",
            "question": "fused softmax 的 speedup 来自减少哪几次 HBM 读写，tail mask 为什么不能省？",
        },
        {
            "node": "Context parallel and long context",
            "what_to_search": "rope, yarn, ring_attention, cp_size, attn_comm_bytes",
            "what_to_inspect": "mini_infra/megatron/core/context_parallel/rope.py; mini_infra/megatron/core/context_parallel/yarn.py; mini_infra/megatron/core/context_parallel/ring_attention.py",
            "question": "CP 省下的 attention 显存何时抵不过 ring KV 通信？",
        },
        {
            "node": "MoE router to all-to-all",
            "what_to_search": "top_k, capacity_factor, router_entropy, alltoall_ms, overflow",
            "what_to_inspect": "mini_infra/megatron/core/transformer/moe/router.py; mini_infra/megatron/core/transformer/moe/capacity.py; mini_infra/megatron/core/transformer/moe/alltoall.py",
            "question": "router collapse、capacity overflow 与 EP all-to-all 慢分别应该看哪个指标？",
        },
        {
            "node": "Quantization and speculative serving",
            "what_to_search": "awq, fp8, kv_int8, calibration, ngram, acceptance, rollback",
            "what_to_inspect": "mini_infra/vllm/quant/awq_loader.py; mini_infra/sglang/quant/kv_int8.py; mini_infra/vllm/spec_decode/draft_runner.py",
            "question": "量化和 speculative decoding 分别改变内存、准确率和尾延迟的哪个边界？",
        },
    ]
)
