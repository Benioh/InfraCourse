# L08 Patch · 把 JSONL 写成 Megatron `.bin/.idx`

## 你要交付什么

在 `patch/starter/megatron_bin.py` 里实现两个接口：

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

`<prefix>.bin` 保存所有样本 token id 的连续二进制拼接。`<prefix>.idx` 保存最小索引元数据。

idx 文件格式，小端：

```text
magic        = b"MGTRNIDX"
version      = uint32 = 1
dtype_code   = uint8  ({1: int32, 2: uint16, 3: int64})
n_samples    = uint64
total_tokens = uint64
offsets[N]   = uint64  # 每条样本在 bin 中的起始字节
lengths[N]   = uint64  # token 数
```

补丁规模目标：60 到 110 行 Python。

## 不变量

1. dtype 只支持 `int32`、`uint16`、`int64`。
2. JSONL 空行、缺失 `text` 字段、空字符串 text 和空 token 序列都要跳过。
3. token id 必须是非负整数，且不能超过当前 dtype 的上限。
4. `offsets[i]` 必须等于前 i 条样本 token 数之和乘 dtype bytes。
5. `lengths[i]` 保存 token 个数，不保存字节数。
6. `.bin` 文件大小必须等于 `total_tokens * dtype_bytes`。
7. `IndexedDataset(prefix)` 必须校验 magic 和 version。
8. `IndexedDataset[i]` 返回 `list[int]`，与写入时 tokenizer 输出一致。
9. `IndexedDataset.__getitem__` 不能全量加载 `.bin`，应使用 `np.memmap`。
10. 下标越界应抛 `IndexError`。

## 怎么验证

```bash
make patch-test M=l07_dataset_megatron_bin
```

6 个测试，全是 CPU 友好：

| 测试 | 验证 |
|---|---|
| `test_writes_bin_idx_pair` | `.bin/.idx` 存在，summary 中样本数和 dtype 正确 |
| `test_idx_header_magic_version` | idx magic、version 和 dtype code 正确 |
| `test_roundtrip_get_sample` | `IndexedDataset[i]` 与 tokenizer 输出一致 |
| `test_total_tokens_matches_concat` | `.bin` 字节数和 total tokens 一致 |
| `test_empty_lines_skipped` | 空文本、缺失 text 和空行跳过 |
| `test_dtype_uint16_overflow_raises` | `uint16` token 越界时报错 |

## 实现提示

- 参考解使用 `numpy.asarray(tokens, dtype=np_dtype).tobytes()` 写 `.bin`。
- idx header 可以用 `struct.pack("<I", VERSION)`、`struct.pack("<B", dtype_code)` 和 `struct.pack("<Q", value)`。
- offsets 和 lengths 可以用 `np.asarray(values, dtype=np.uint64).tobytes()` 写入。
- 读取 idx 时先读固定长度 header，再用 `np.frombuffer(...).copy()` 拿到 offsets 和 lengths。
- mmap `.bin` 时 dtype 必须来自 idx 中的 dtype code。

## 写完之后你能做什么

- 把 demo JSONL 转成后续 Megatron pretrain 可以引用的 data prefix。
- 解释 `.bin/.idx` 的样本边界、dtype 和 mmap 读取合同。
- 看懂真实 Megatron `preprocess_data.py` 和 `IndexedDatasetBuilder` 的主路径。
