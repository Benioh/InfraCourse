# Source Reading Card：L37 CUDA IPC Weight Sync

## 主路径

1. `labs/l32.5_ipc_weight_sync/patch/reference/ipc_weight_sync.py`：handle meta、共享 storage、rank gather 和 final flush。
2. `labs/l32.5_ipc_weight_sync/patch/tests/test_patch.py`：7 个 CPU 合同行为测试。
3. `github_repo/sglang/python/sglang/srt/model_executor/model_runner.py`：SGLang 在线 tensor update、unwrap 和 load_weights。
4. `github_repo/sglang/python/sglang/srt/utils/common.py`：MultiprocessingSerializer 的 ForkingPickler 包装。
5. `github_repo/sglang/python/sglang/srt/utils/patch_torch.py`：CUDA tensor reduction 的 device 修正。
6. `github_repo/sglang/python/sglang/srt/weight_sync/tensor_bucket.py`：flattened bucket 和 metadata 重建。
7. `github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`：SLiME bucket、gather_object、engine update 和 ipc_collect。

## 每段要得到的结论

| 文件 | 读完后要能说明 |
|---|---|
| `ipc_weight_sync.py` | 为什么 blob 小、data_ptr 一致、rank 0 收齐后才能返回 |
| `test_patch.py` | CPU patch 覆盖了哪些合同，没覆盖哪些 CUDA IPC 生产边界 |
| `model_runner.py` | SGLang 如何按 TP rank unwrap LocalSerializedTensor 并写模型 |
| `common.py` | ForkingPickler 如何承接 torch tensor reduction |
| `patch_torch.py` | 为什么跨进程 device 信息需要重写 |
| `tensor_bucket.py` | flattened tensor 和 metadata 如何表达多个参数 |
| `update_weight_from_tensor.py` | SLiME 如何 pause、分桶、serialize、gather、remote update 和 ipc_collect |

## 自检

- 为什么 handle bytes 不能接近 tensor data？
- 为什么 `torch.equal` 不足以证明共享 storage？
- `LocalSerializedTensor.values[rank]` 的 rank 顺序错了会怎样？
- 本讲目录为什么没有 dedicated smoke target？
