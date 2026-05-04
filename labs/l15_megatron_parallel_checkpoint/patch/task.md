# L05.8 Patch · Megatron-shaped Distributed Checkpointing

## 你要交付什么

实现 `mini_infra/megatron/training/checkpointing.py` 的增强版：保存 checkpoint 时同时保存 model、optimizer、scheduler、parallel state；加载 checkpoint 时校验并行状态兼容性。

```python
def save_checkpoint(output_dir, iteration, model_state, optimizer_state, scheduler_state, parallel_state) -> dict: ...
def load_checkpoint(checkpoint_dir, expected_parallel_state=None, strict=True) -> dict: ...
```

补丁规模目标：60-100 行。

## 不变量

1. checkpoint payload 必须包含 format、iteration、model/optimizer/scheduler/parallel state。
2. 必须写 `latest_checkpointed_iteration.txt`。
3. load 时必须从 latest marker 找到对应 checkpoint 文件。
4. `strict=True` 时并行状态不匹配要抛 `CheckpointError`。
5. `strict=False` 时不抛错，但必须返回 warnings。

## 怎么验证

```bash
make patch-test M=l15_megatron_parallel_checkpoint
```

## 写完之后你能做什么

- 解释为什么 Megatron resume 不只是加载 model weights。
- 解释 scheduler/global step 和 parallel state 为什么必须进 checkpoint。
- 定位 checkpoint shape mismatch / TP mismatch 的第一层证据。
