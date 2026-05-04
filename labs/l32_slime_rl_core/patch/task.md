# L11 Patch · SLiME Weight Sync Coordinator

## 你要交付什么

实现 RL 训练中**训练侧 ↔ 推理侧 weight sync** 的核心逻辑：

```python
class WeightSyncCoordinator:
    def __init__(self, train_state_provider, inference_state_setter): ...
    def sync(self) -> dict:
        """从 train 拉新 weights，写入 inference；返回 stats:
        {bytes_synced, num_tensors, mismatched_keys}"""
```

**禁止** 用 `torch.distributed` 的真实 broadcast（本关 CPU 单机模拟）。
**允许** 普通 dict 操作 + tensor copy。

补丁规模目标：30–60 行。

## 接口契约

```python
train_state = {
    "embed.weight": torch.randn(100, 16),
    "layer.weight": torch.randn(8, 16),
}
inference_state = {
    "embed.weight": torch.zeros(100, 16),
    "layer.weight": torch.zeros(8, 16),
}

coord = WeightSyncCoordinator(
    train_state_provider=lambda: train_state,
    inference_state_setter=lambda new_state: inference_state.update(new_state),
)

stats = coord.sync()
# inference_state 现在等于 train_state
# stats = {"bytes_synced": ..., "num_tensors": 2, "mismatched_keys": []}
```

## 不变量

1. sync 后 inference state 的每个 tensor 在数值上等于 train state（torch.equal）。
2. shape 不匹配的 key 不写入，记录到 `mismatched_keys`。
3. dtype 不强制转换；如果 dtype 不同也算 mismatch。
4. inference state 中存在但 train 没有的 key 保持不变（不删）。
5. train 中存在但 inference 没有的 key 跳过（不强制创建，记 mismatch）。
6. bytes_synced = sum(t.numel() * t.element_size()) for synced tensors.

## 怎么验证

```bash
make patch-test M=l32_slime_rl_core
```

5 个测试：

| 测试 | 验证 |
|---|---|
| `test_basic_sync` | 数值与 train 完全相等 |
| `test_shape_mismatch_skipped` | shape 不一致进 mismatched_keys |
| `test_dtype_mismatch_skipped` | dtype 不一致也算 mismatch |
| `test_extra_train_keys_skipped` | train 多余 key 不创建到 inference |
| `test_stats_correct` | bytes_synced / num_tensors 数字对 |

## 写完之后你能做什么

- 解释 SLiME / OpenRLHF 的 weight sync 机制（NCCL broadcast 跨集群）。
- 在 Capstone Stage C 实现训练 ↔ 推理 endpoint 的 weight 推送。
- 看懂 sharded weight sync（每 rank 持有一部分）的扩展实现。
