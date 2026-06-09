# 源码带读：L37 CUDA IPC Weight Sync

这份带读按“patch 合同 -> SGLang 接收 -> SLiME 发送 -> serializer/bucket”的顺序走。第一次读时先把 handle、rank、bucket 和 flush 串起来，再回头看性能分支。

## 0. 源码地图

```text
labs/l32.5_ipc_weight_sync/patch/starter/ipc_weight_sync.py
labs/l32.5_ipc_weight_sync/patch/reference/ipc_weight_sync.py
labs/l32.5_ipc_weight_sync/patch/tests/test_patch.py

github_repo/sglang/python/sglang/srt/model_executor/model_runner.py
github_repo/sglang/python/sglang/srt/utils/common.py
github_repo/sglang/python/sglang/srt/utils/patch_torch.py
github_repo/sglang/python/sglang/srt/weight_sync/tensor_bucket.py

github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
```

## 1. Patch 合同

文件：[patch/starter/ipc_weight_sync.py](patch/starter/ipc_weight_sync.py)

先看第 26-38 行。`IPCStoragePool` 保存 tensor 引用，并返回 uuid handle；这是 CPU 模拟里共享 storage 的来源。

再看第 60-79 行。`serialize_handle` 的合同是只保存 handle 和 tensor 元数据，不保存 tensor data。第 82-92 行的 `deserialize_handle` 反过来按 handle 取回 tensor。

最后看第 95-156 行。`LocalSerializedTensor`、`gather_handles_to_rank0` 和 `update_weights_from_tensor` 分别对应真实系统里的 rank handle 列表、gather_object 和 SGLang 侧权重替换。

文件：[patch/reference/ipc_weight_sync.py](patch/reference/ipc_weight_sync.py)

第 39-48 行是 handle meta 序列化；第 51-53 行是从 pool 取回 tensor；第 64-76 行模拟 rank 0 gather；第 79-91 行完成 state 替换和最后 flush。

文件：[patch/tests/test_patch.py](patch/tests/test_patch.py)

按顺序读七个测试：

- 第 22-29 行：序列化 blob 不能接近 tensor data 大小。
- 第 32-39 行：反序列化后必须共享 `data_ptr()`。
- 第 42-49 行：共享 tensor 数值也要一致。
- 第 52-70 行：rank 0 gather 的返回语义。
- 第 73-81 行：`LocalSerializedTensor.get` 按 rank 取 tensor。
- 第 84-107 行：update 后 inference state 被替换。
- 第 110-139 行：`flush_cache=True` 只在最后清 pool。

## 2. SGLang 接收路径

文件：[github_repo/sglang/python/sglang/srt/model_executor/model_runner.py](../../github_repo/sglang/python/sglang/srt/model_executor/model_runner.py)

第 1932-1940 行是在线 tensor update 入口。先调用 `monkey_patch_torch_reductions()`，再分流 flattened bucket 和普通 named tensors。

第 1944-1951 行把每个 tensor unwrap。若它是 `LocalSerializedTensor`，就按当前 TP rank 取出对应 handle。第 1952-1961 行根据 `load_format` 选择 loader，最后写入模型。

第 3414-3417 行是 `_unwrap_tensor`；第 3453-3461 行定义真实 `LocalSerializedTensor`，它也用 `values[rank]` 取 serialized tensor。

## 3. Serializer 和 Torch Reduction

文件：[github_repo/sglang/python/sglang/srt/utils/common.py](../../github_repo/sglang/python/sglang/srt/utils/common.py)

第 2179-2201 行展示 `MultiprocessingSerializer.serialize` 使用 `ForkingPickler`，可选择输出 base64 字符串。第 2203-2215 行是反序列化入口。

文件：[github_repo/sglang/python/sglang/srt/utils/patch_torch.py](../../github_repo/sglang/python/sglang/srt/utils/patch_torch.py)

第 40-51 行 patch torch 的 reduce/rebuild tensor 逻辑。第 71-85 行把 device 表示改成 UUID，再在 rebuild 时解析回本进程可见 device。第一次阅读不用展开 PyTorch reduction 的全部 14 元组，先记住它解决跨进程 device 映射。

## 4. Flattened Bucket

文件：[github_repo/sglang/python/sglang/srt/weight_sync/tensor_bucket.py](../../github_repo/sglang/python/sglang/srt/weight_sync/tensor_bucket.py)

第 7-16 行定义 metadata：name、shape、dtype、start/end、numel。第 49-72 行把多个 tensor flatten 为一个 uint8 buffer。第 90-105 行按 metadata 切片并 reshape 回原参数。

这解释了为什么 SLiME co-locate 路径常常按 bucket 发送，而不是每个参数单独发。

## 5. SLiME 发送路径

文件：[github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py](../../github_repo/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py)

第 137-154 行开始一次 update：递增 version，rank 0 暂停 generation 并 flush cache。第 156-166 行逐 chunk 发送、等待 refs，并调用 `torch.cuda.ipc_collect()`。

第 209-220 行处理没有 colocated engine 的 rank。第 234-244 行构造 flattened bucket，并用 `MultiprocessingSerializer` 序列化。第 245-253 行用 `gather_object` 收集每个 rank 的 serialized tensors。第 255-267 行由 src rank 调用 rollout engine 的 `update_weights_from_tensor`。

## 可以先跳过

- SGLang custom weight loader 的所有分支。
- SLiME 里量化前后处理和 distributed update 路径。
- PyTorch reduction tuple 的每个底层字段。先把 handle、rank 和生命周期读通。

## 自检问题

1. patch 的 `IPCStoragePool` 如何模拟共享 storage？
2. 为什么 `LocalSerializedTensor.values` 要按 rank 保存多个 blob？
3. SGLang 在哪里从 LocalSerializedTensor 取出当前 TP rank 的 tensor？
4. SLiME 为什么要把多个 tensor flatten 成 bucket？
5. 真 CUDA IPC 验证还要补哪些 CPU patch 没覆盖的证据？
