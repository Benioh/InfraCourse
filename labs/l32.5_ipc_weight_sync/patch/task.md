# L37 Patch · CUDA IPC Weight Sync

## 你要交付什么

实现 verl/slime co-locate 路径下 `update_weights_from_tensor` 的核心机制——
**用 handle tuple 共享显存而不是搬数据**。

```python
def serialize_handle(tensor: torch.Tensor, pool: IPCStoragePool) -> bytes: ...
def deserialize_handle(blob: bytes, pool: IPCStoragePool) -> torch.Tensor: ...

@dataclass
class LocalSerializedTensor:
    values: list[bytes]
    def get(self, rank: int, pool: IPCStoragePool) -> torch.Tensor: ...

def gather_handles_to_rank0(
    local_blob: bytes, rank: int, world_size: int, group: dict,
) -> list[bytes] | None: ...

def update_weights_from_tensor(
    named_handles: list[tuple[str, LocalSerializedTensor]],
    inference_state: dict[str, torch.Tensor],
    tp_rank: int,
    pool: IPCStoragePool,
    flush_cache: bool = False,
) -> None: ...
```

`IPCStoragePool`（已提供）是 GPU IPC 存储的 CPU 模拟：把 tensor 注册进 pool 拿一个 uuid handle；
反序列化时按 handle 查 pool 返回的 tensor 与原 tensor 共享 storage。

**禁止** 把 tensor data 塞进 bytes（那就退化成普通 copy 了）。
**允许** `pickle` / `uuid` / `dataclasses`。

补丁规模目标：60–110 行 Python。

## 接口契约

```python
pool = IPCStoragePool()

# Source side (FSDP TP rank)
src_tensor = torch.randn(1024, 1024)
blob = serialize_handle(src_tensor, pool)
assert len(blob) < 1024 * 1024 * 4  # << tensor data size

# Cross-process (or just within process for our sim)
recv_blob = blob

# Destination side (SGLang TP rank)
dst_tensor = deserialize_handle(recv_blob, pool)
assert dst_tensor.data_ptr() == src_tensor.data_ptr()  # share storage
assert torch.equal(dst_tensor, src_tensor)
```

Multi-rank gather + LocalSerializedTensor:

```python
# Each FSDP TP rank serializes its own (already-aggregated) tensor:
group = {}  # simulates distributed group state
local_blob = serialize_handle(tensor, pool)
gathered = gather_handles_to_rank0(local_blob, rank=0, world_size=4, group=group)
# rank 0 gets [b0, b1, b2, b3]; rank != 0 gets None.

# Pack into LocalSerializedTensor (one per parameter):
lst = LocalSerializedTensor(values=gathered)

# SGLang Engine side: each SGLang TP rank reconstructs its own:
my_tensor = lst.get(my_tp_rank, pool)
```

End-to-end weight sync:

```python
update_weights_from_tensor(
    named_handles=[("layer.weight", lst1), ("layer.bias", lst2)],
    inference_state=engine_state,
    tp_rank=0,
    pool=pool,
    flush_cache=True,  # last call → free pool
)
```

## 不变量

1. `serialize_handle(t, pool)` 返回的 bytes **不包含 tensor data**；可以包含 shape/dtype/device/stride/handle。
2. `deserialize_handle(serialize_handle(t, pool), pool)` 与 `t` **共享 storage**（`data_ptr() == t.data_ptr()`）。
3. `gather_handles_to_rank0`：`rank != 0` 总是返回 `None`；`rank == 0` 在 `len(group) < world_size` 时返回 `None`，群组凑齐后返回长度 `world_size` 的列表（这模拟"rank 0 先注册再轮询"的行为）。
4. `LocalSerializedTensor.get(rank, pool)` 等价于 `deserialize_handle(self.values[rank], pool)`。
5. `update_weights_from_tensor` 只在 **最后一个参数**（即 `tensor_index == len(named_handles) - 1`）的 `flush_cache=True` 触发 pool 清空。
6. `update_weights_from_tensor` 调用后 `inference_state[name]` 严格等于源 tensor。

## 怎么验证

```bash
make patch-test M=l32.5_ipc_weight_sync
```

7 个测试：

| 测试 | 验证 |
|---|---|
| `test_serialize_returns_handle_not_data` | bytes 长度远小于 tensor 字节数 |
| `test_deserialize_shares_storage` | 反序列化后 data_ptr 一致 |
| `test_handle_round_trip_preserves_values` | torch.equal 一致 |
| `test_gather_only_rank_0_has_full_list` | rank!=0 → None |
| `test_local_serialized_tensor_get_by_rank` | 4 rank 各 get 自己 |
| `test_update_weights_replaces_inference_state` | 集成调用后 state 替换 |
| `test_flush_cache_only_on_last_tensor` | flush 只在最后 |

## 卡住怎么办

1. 跑 `notebooks/n22_weight_sync_handle_tuple.ipynb` 把 IPC handle 的语义打牢。
2. `make patch-hint` 看 TODO；`make patch-show-solution` 看参考解。

## 写完之后你能做什么

- 解释 verl `LocalSerializedTensor` / `MultiprocessingSerializer` 的真实实现每一行。
- 区分 `from_disk` / `from_distributed` / `from_tensor` 三种 weight sync 接口的取舍。
- 看懂 SGLang `monkey_patch_torch_reductions`、`reduce_tensor` 返回的 14 元组是什么。
- 估算 700B MoE 在 FSDP→SGLang 切换时 handle 序列化 / 反序列化的时间预算。
