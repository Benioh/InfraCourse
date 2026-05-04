# L03.5 Patch · 把 JSONL 写成 Megatron `.bin/.idx`

## 你要交付什么

```python
def text_to_megatron_bin(
    jsonl_path: Path,
    tokenizer: Callable[[str], list[int]],
    output_prefix: Path,
    dtype_str: str = "int32",
) -> dict: ...

class IndexedDataset:
    def __init__(self, prefix: Path): ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> list[int]: ...
```

文件格式（与 Megatron `IndexedDataset` 同构、简化版）：

`<prefix>.bin`：所有样本的 token id 顺序拼接，按 `dtype_str` 序列化。

`<prefix>.idx`（小端）：
```
magic       = b"MGTRNIDX"      # 8 字节
version     = uint32 = 1
dtype_code  = uint8  ({1: int32, 2: uint16, 3: int64})
n_samples   = uint64
total_tokens= uint64
offsets[N]  = uint64  # 每条样本在 bin 中的起始字节
lengths[N]  = uint64  # token 数（不是字节数）
```

补丁规模目标：60-100 行。

## 不变量

1. 跳过 `text` 字段为空字符串或缺失的行；返回的 `n_samples` 必须等于实际写入条数。
2. dtype `int32 / uint16 / int64` 三选一；token 越界要抛 `ValueError`（uint16 时 ≥ 65536 越界）。
3. `offsets[i]` 必须等于前 i 条 lengths 累加 × dtype 字节数。
4. `IndexedDataset[i]` 返回 `list[int]`，与写入时的 token list 完全一致。
5. 必须支持 mmap 读（`np.memmap` 即可）；不能在 `__getitem__` 里全量 load。

## 怎么验证

```bash
make patch-test M=l07_dataset_megatron_bin
```

## 写完之后你能做什么

- 在 L04 用真实 FineWeb-Edu / WikiText 启动 Megatron pretrain
- 解释 Megatron 多 worker dataloader 为什么可以 mmap 共享
- 看懂 `tools/preprocess_data.py` 的 chunk pipeline
