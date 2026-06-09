# L36 Patch · SLiME Weight Sync Coordinator

## 你要交付什么

实现一个本地版 `WeightSyncCoordinator`，模拟 RL 训练中 actor 权重从 train side 同步到 inference / rollout side 的最小合同。它从训练侧读取 `state_dict`，只复制 inference 侧已经存在且 shape、dtype 都匹配的 tensor，并返回同步统计。

```python
class WeightSyncCoordinator:
    def __init__(
        self,
        train_state_provider,
        inference_state_setter,
        inference_state_provider=None,
    ) -> None: ...

    def sync(self) -> dict:
        """返回 bytes_synced、num_tensors、mismatched_keys。"""
```

本关禁止使用真实 `torch.distributed.broadcast`。允许普通 dict 操作和 PyTorch tensor copy。补丁规模目标是 30 到 60 行。

## 接口合同

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
    inference_state_provider=lambda: inference_state,
)

stats = coord.sync()
assert torch.equal(inference_state["embed.weight"], train_state["embed.weight"])
assert stats["num_tensors"] == 2
```

## 不变量

1. key 存在、shape 一致、dtype 一致时才同步。
2. shape 不一致的 key 不写入 inference，记录到 `mismatched_keys`。
3. dtype 不一致也算 mismatch，不自动 cast。
4. train 多出的 key 不创建到 inference，记录到 `mismatched_keys`。
5. inference 多出的 key 保持不变，不删除。
6. `bytes_synced` 只统计真正同步的 tensor。
7. 同步 tensor 要 `detach().clone()`，避免 inference 继续引用 train tensor。

## 验证命令

```bash
make patch-test M=l32_slime_rl_core
```

参考实现验收：

```bash
IMPL=reference make patch-test M=l32_slime_rl_core
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_basic_sync` | inference 数值与 train 完全一致 |
| `test_shape_mismatch_skipped` | shape 不一致的 key 被跳过 |
| `test_dtype_mismatch_skipped` | dtype 不一致不会自动转换 |
| `test_extra_train_keys_skipped` | train 多余 key 不创建到 inference |
| `test_stats_correct` | `bytes_synced` 和 `num_tensors` 只统计 accepted tensors |

## 写完后要能解释

- 为什么本地 patch 不等于 NCCL weight sync。
- 为什么 dtype 转换应在发送方显式做，而不是在同步函数里静默 cast。
- 为什么真实 SLiME 还需要 Ray 协调、engine lock、bucket、metadata 和 weight version。
