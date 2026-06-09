# Source Reading Card：L34 CUDA Graph Cache + Memory Savor

## 主路径

1. `labs/l30.5_cuda_graph_savor/patch/starter/cuda_graph_cache.py`：学生要实现的 cache 和 handle 状态机。
2. `labs/l30.5_cuda_graph_savor/patch/reference/cuda_graph_cache.py`：最小正确公式和状态更新。
3. `labs/l30.5_cuda_graph_savor/patch/tests/test_patch.py`：8 个行为合同。
4. `github_repo/Megatron-LM/megatron/core/transformer/cuda_graphs.py`：真实 CUDA Graph metadata 和 buffer reuse pool。
5. `github_repo/Megatron-LM/megatron/training/arguments.py`：生产 CUDA Graph、decode-only 和 RL KV cache 配置边界。
6. `github_repo/slime/slime/utils/memory_utils.py`：显存 snapshot 和清理入口。

## 阅读顺序

1. 先看 reference 的 `GraphCache`，确认 key、capture 和 replay。
2. 再看 reference 的 `MemorySavor`，确认 handle、metadata、pop 和 bytes。
3. 然后看 tests，把每个断言对应到一个行为合同。
4. 最后读 Megatron 的 `ArgMetadata` 与 `TensorReusePool`，理解真实 graph 为什么记录地址和强引用 buffer。

## 自检

- 我能否解释 CPU patch 验证了什么，没验证什么？
- 我能否说明真实 CUDA Graph 为什么需要 static buffer？
- 我能否写出 co-locate 的 pause/train/offload/resume 顺序？
- 我能否列出 GPU 验证必须记录的地址、timing 和显存证据？
