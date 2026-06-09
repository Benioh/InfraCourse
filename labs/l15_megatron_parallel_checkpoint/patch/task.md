# L16 Patch · Megatron-shaped Distributed Checkpointing

## 你要交付什么

实现 `patch/starter/checkpointing.py` 中的两个函数：

```python
def save_checkpoint(
    output_dir: Path,
    iteration: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any],
    parallel_state: dict[str, Any],
) -> dict[str, str]: ...

def load_checkpoint(
    checkpoint_dir: Path,
    expected_parallel_state: dict[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]: ...
```

补丁规模目标：40 到 80 行。

## 不变量

1. `iteration < 0` 时抛 `CheckpointError`。
2. checkpoint payload 必须包含 `format`、`iteration`、`model_state`、`optimizer_state`、`scheduler_state`、`parallel_state`。
3. 保存时必须写 `latest_checkpointed_iteration.txt`。
4. load 时必须从 latest marker 找到对应 checkpoint 文件。
5. payload `format` 不匹配时抛 `CheckpointError`。
6. `strict=True` 时 parallel_state 不匹配要抛 `CheckpointError`。
7. `strict=False` 时不抛错，但必须把 mismatch 写入 `warnings`。
8. load 返回的 payload 必须包含 `checkpoint_path` 和 `warnings`。

## 怎么验证

```bash
make patch-test M=l15_megatron_parallel_checkpoint
```

通过后跑一次 drill：

```bash
IMPL=reference bash labs/l15_megatron_parallel_checkpoint/scripts/run_drill.sh l16_validation
```

## 写完之后你能做什么

- 解释为什么 Megatron resume 需要 optimizer、scheduler 和 parallel_state。
- 定位 missing marker、format mismatch、TP/PP/EP mismatch 的第一层证据。
- 区分 strict resume、non-strict 诊断、checkpoint 转换和 finetune 权重加载。
