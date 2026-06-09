# Debug Checklist：L34 CUDA Graph Cache + Memory Savor

## 1. 固定现场

- 记录命令、配置、git commit、PyTorch/CUDA 版本、GPU 型号、batch shape 和随机种子。
- 保存 patch-test 输出、notebook 结果、graph timing、`data_ptr()` 记录和 memory snapshot。
- 区分当前运行是 CPU patch、notebook 语义演示、单卡 CUDA Graph，还是多进程 co-locate。

## 2. 检查 GraphCache

| 检查项 | 证据 | 失败时的判断 |
|---|---|---|
| cache key | args/kwargs shape、dtype、scalar | key 太粗会错用 graph，key 太细会频繁 miss |
| capture_count | 首次 shape 是否只 capture 一次 | 初始化或 key 生成有问题 |
| replay_count | 同 shape 是否命中 replay | shape/dtype/scalar 不稳定 |
| GPU 地址 | static buffer `data_ptr()` 是否不变 | 真实 replay 可能读写错误地址 |
| timing | graph replay 与 eager 对比 | CPU patch 不能证明性能收益 |

## 3. 检查 MemorySavor

- pause 后 handle 是否存在。
- paused bytes 是否等于池内数据 `numel * element_size` 总和。
- resume 是否返回相同数据。
- resume 后 handle 是否从池中移除。
- 真实 GPU 场景中，虚拟地址和物理页映射是否有独立证据。

## 4. 检查 co-locate 切换

1. rollout 完成后是否先 pause rollout 大块显存。
2. training weights / optimizer / activation 是否在 pause 后才进入高峰。
3. training 结束后是否先 offload，再 resume rollout。
4. suspend/resume 时是否保留或重建 CUDA Graph，配置是否清楚。
5. 每个阶段是否有 memory snapshot 和 graph replay 状态。

## 5. 沿源码主路径复查

- `labs/l30.5_cuda_graph_savor/patch/starter/cuda_graph_cache.py`：学生实现的 GraphCache 和 MemorySavor。
- `labs/l30.5_cuda_graph_savor/patch/reference/cuda_graph_cache.py`：最小正确状态机。
- `labs/l30.5_cuda_graph_savor/patch/tests/test_patch.py`：8 个行为合同。
- `github_repo/Megatron-LM/megatron/core/transformer/cuda_graphs.py`：真实 graph metadata、buffer pool 和地址记录。
- `github_repo/slime/slime/utils/memory_utils.py`：显存快照和清理工具。

## 6. 常见错误判断

- 把 CPU patch 通过写成 CUDA Graph 已加速。
- 只看 shape，不记录 GPU buffer 地址。
- pause 后没有检查 bytes，resume 后没有检查 handle 移除。
- co-locate OOM 时只看单阶段显存，不看切换顺序。
- graph cache bucket 配置和实际请求 batch 分布不匹配。

## 7. 结束条件

- 问题能被一个最小 shape 或一条命令复现。
- capture/replay 计数、paused bytes、地址和 timing 已保存。
- 能指出源码中 key、metadata、handle 或 memory snapshot 的状态变化位置。
- 结论写入 `rl_rollout_template.md`，并区分 CPU 已验证、GPU 待验证和生产风险。
