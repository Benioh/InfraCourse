# L32.5 · CUDA IPC Weight Sync（真正的机制）

> L32 教的是"weight sync 的语义"——dict 怎么按 shape/dtype 校验后写入。
> L32.5 教的是"weight sync 的真正机制"——**handle tuple 共享**：序列化的不是数据，是指针；
> 只要 IPC handle 传到对端，对端就能用同一块 GPU 显存里的 tensor 完成更新。
>
> 这是 verl `update_weights_from_tensor` 与 slime `_update_converted_params_from_tensor`
> 的核心实现。CPU 上我们用一个共享 `StoragePool` 模拟，但 API 形状与真实代码一致。

## 真实场景

参考 [RL 系统深思：深入理解权重更新机制](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/sys-design/readme-1.md)。
verl 的 co-locate 路径下，FSDP TP=4、SGLang TP=2 时，每个 FSDP rank 把本 rank 的分片
聚合成 `[1024, 1024]` 完整 tensor，序列化得到 **handle tuple**（CUDA IPC handle + 元信息），
gather 到 TP=0 后跨进程传递给 SGLang Engine。SGLang 每个 TP rank 反序列化重建 tensor，
**和 FSDP 共享同一块 GPU 显存**，没有数据搬运。

## 闭环

```bash
cat labs/l32.5_ipc_weight_sync/patch/task.md
$EDITOR labs/l32.5_ipc_weight_sync/patch/starter/ipc_weight_sync.py
make patch-test M=l32.5_ipc_weight_sync
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_serialize_returns_handle_not_data` | 序列化 bytes < tensor data 字节数（不能藏数据） |
| `test_deserialize_shares_storage` | 反序列化后 `data_ptr()` 与原 tensor 相同 |
| `test_handle_round_trip_preserves_values` | 数值层面完全一致 |
| `test_gather_only_rank_0_has_full_list` | 非 rank-0 返回 None |
| `test_local_serialized_tensor_get_by_rank` | 多 rank LST 按 rank 取 |
| `test_update_weights_replaces_inference_state` | 集成：调一次 update，inference state 完整替换 |
| `test_flush_cache_only_on_last_tensor` | 最后一个 tensor 才 free pool |

## 卡住怎么办

1. 先看 `notebooks/n22_weight_sync_handle_tuple.ipynb` 把 handle 共享的图画一遍。
2. `make patch-hint M=l32.5_ipc_weight_sync` 看 TODO 与提示。
3. `make patch-show-solution M=l32.5_ipc_weight_sync` 看参考解。

## 写完之后你能做什么

- 解释 verl `_preprocess_tensor_for_update_weights` → `MultiprocessingSerializer.serialize`
  → `dist.gather_object` → `update_weights_from_tensor` 整条链路每一步在做什么。
- 区分 `update_weights_from_disk` / `update_weights_from_distributed` /
  `update_weights_from_tensor` 三种接口的取舍（co-locate vs disaggregate、动态扩缩容代价）。
- 看懂 SGLang `LocalSerializedTensor`、`MultiprocessingSerializer`、`monkey_patch_torch_reductions` 的实现。
- 给 SGLang RL 调试日志加上 handle 大小 / pool 占用监控。

## 配套源码研读（可选）

- `github_repo/verl/verl/workers/sharding_manager/fsdp_sglang.py` — co-locate update_weights
- `github_repo/sglang/python/sglang/srt/model_executor/model_runner.py::update_weights_from_tensor`
- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/sys-design/readme-1.md` — 整篇必读
