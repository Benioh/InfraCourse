# 源码带读：L34 CUDA Graph Cache + Memory Savor

这份带读按“patch 合同 -> pytest 边界 -> Megatron CUDA Graph -> SLiME memory utility”的顺序走。读源码时先抓住三个状态：cache key、captured function、paused handle。

## 0. 源码地图

```text
labs/l30.5_cuda_graph_savor/patch/starter/cuda_graph_cache.py
labs/l30.5_cuda_graph_savor/patch/reference/cuda_graph_cache.py
labs/l30.5_cuda_graph_savor/patch/tests/test_patch.py

github_repo/Megatron-LM/megatron/core/transformer/cuda_graphs.py
github_repo/Megatron-LM/megatron/training/arguments.py
github_repo/slime/slime/utils/memory_utils.py
```

## 1. Patch starter：确认当前真实接口

文件：[patch/starter/cuda_graph_cache.py](patch/starter/cuda_graph_cache.py)

第 20-36 行定义 `GraphCache` 的状态和 `_shape_key`。注意当前接口从 args/kwargs 的 tensor shape、dtype 和 scalar 值生成 key，旧版按 `bs` 入参缓存的描述已经不适用。

第 38-53 行是 `capture_or_replay` 的 TODO。首次 key miss 时保存函数引用并计 capture；key hit 时计 replay，并调用已保存的函数。

第 56-74 行是 `MemorySavor.pause`，第 76-91 行是 `resume`、`total_paused_bytes` 和 `is_paused`。这一段练的是 handle 池、metadata 和恢复后移除。

## 2. Patch reference：最小正确实现

文件：[patch/reference/cuda_graph_cache.py](patch/reference/cuda_graph_cache.py)

第 10-20 行保存 `_graphs`、计数器和 shape key。第 22-31 行把 args/kwargs 变成 key，并区分 capture 与 replay。

第 39-47 行展示 pause 如何生成 handle 并保存 metadata。第 49-57 行展示 resume 如何 `pop` handle、返回 clone，并按池内数据统计 bytes。

## 3. Patch tests：8 个行为合同

文件：[patch/tests/test_patch.py](patch/tests/test_patch.py)

第 22-52 行覆盖 GraphCache 的前三个合同：首次 capture、第二次 replay、不同 shape 新 capture。第 55-65 行检查 replay 输出与 eager 一致。

第 68-104 行覆盖 MemorySavor：pause 后 handle 存在、resume 数据相等、paused bytes 统计正确、resume 后 handle 消失且 bytes 归零。

## 4. Megatron CUDA Graph 对照

文件：[github_repo/Megatron-LM/megatron/core/transformer/cuda_graphs.py](../../github_repo/Megatron-LM/megatron/core/transformer/cuda_graphs.py)

第 84-98 行展示 capture 状态标记。第 118-130 行定义 cudagraph buffer metadata，记录输入输出、use count 和复用计数。第 138-148 行的 `ArgMetadata` 会保存 tensor shape、dtype、device 和 `data_ptr()`，说明真实 graph 关心地址。

第 161-166 行解释 tensor reuse pool 会强引用 buffer，防止 allocator 回收。第 184-201 行展示 pool 如何判断是否 owns 某个 tensor，并按 shape/dtype/device 复用或新建 buffer。

文件：[github_repo/Megatron-LM/megatron/training/arguments.py](../../github_repo/Megatron-LM/megatron/training/arguments.py)

第 1643-1658 行展示启用 CUDA Graph 时会打开相关 RNG tracker，并处理 allocator/NCCL 的兼容风险。第 1918-1926 行说明动态图推理可以配置 capture 的 graph 数量。第 1937-1939 行提供 decode-only CUDA Graph 开关。第 2418-2426 行展示 RL KV cache 管理和 suspend 时是否保留 CUDA Graph 的边界。

## 5. SLiME memory utility

文件：[github_repo/slime/slime/utils/memory_utils.py](../../github_repo/slime/slime/utils/memory_utils.py)

第 11-17 行清理 CUDA cache 和可选 host cache。第 19-29 行读取当前 GPU 的 total/free/allocated/reserved 信息。第 41-49 行按 rank 记录 memory snapshot。这不是 MemorySavor 实现，但它是排查 co-locate 显存切换时最基础的观测入口。

## 可以先跳过的内容

- Megatron CUDA Graph helper 中的完整 capture、warmup、FP8 和 transformer engine 分支。
- 真实 VMM allocator 或 `torch_memory_saver` 的 C++/CUDA 实现。
- 多进程 co-locate 调度和权重同步；它们属于后续 rollout / weight sync 课程。

## 读完后的自检问题

1. 本关 `GraphCache` 的 key 由哪些输入组成？
2. 为什么真实 graph metadata 要记录 `data_ptr()`？
3. `MemorySavor.resume` 为什么要从池中移除 handle？
4. CPU patch 通过后，还需要哪些 GPU 证据才能证明 graph replay 成立？
