# L07 · TorchTitan：Selective Activation Checkpoint

这一讲解决训练系统里常见的 activation 显存问题：Transformer forward 会保存大量中间 activation 给 backward 使用，当 `micro_batch_size`、`seq_len`、`hidden` 或层数增加时，峰值显存可能先由 activation 撑爆。本讲实现一个选择性 wrapper，让 policy 决定哪些直接子模块用 `torch.utils.checkpoint` 重算，哪些保持普通 autograd。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 activation checkpoint 在训练 step、FSDP、compile 和 checkpoint save/load 中的位置。
2. 读 [lecture.md](lecture.md)：理解 checkpoint 的输入、中间状态、输出、代价、RNG 和 policy 边界。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 starter、reference、tests、TorchTitan `apply_ac` 和 smoke 路径读源码。
4. 跑 notebook：[n06_activation_recompute.ipynb](../../notebooks/n06_activation_recompute.ipynb)。
5. 做 quiz：确认显存换计算、选择 attention 的理由、DDP/compile/RNG 边界。
6. 做 patch：实现 `selective_checkpoint_wrap` 和 `attention_only_policy`。
7. 跑 smoke：验证训练循环、metrics、checkpoint 和 resume artifact。
8. 填写 [outputs/training_step_template.md](outputs/training_step_template.md)。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems / activation memory |
| 它解决什么问题 | 训练 step 中哪些子模块可以少存 activation，并在 backward 重算 |
| 它连接哪些指标 | output diff、input grad diff、param grad diff、wrapped child count、CUDA peak memory、step time |
| 它连接哪些源码 | `patch/reference/selective_ckpt.py`、`patch/tests/test_patch.py`、TorchTitan `activation_checkpoint.py`、`parallelize.py` |
| lab 检验什么 | policy 选中的 child 被 wrapper 包住；输出和梯度与裸模型对齐；全 False policy 是 no-op |

## 你会学到什么

- activation checkpoint 与磁盘训练 checkpoint 的区别。
- `torch.utils.checkpoint(..., use_reentrant=False)` 为什么会让 backward 重跑 forward。
- policy 函数如何把“哪些层重算”变成可测试规则。
- wrapper 为什么要持有原 child module，不能复制参数。
- TorchTitan 为什么在并行化和 FSDP 之间固定 activation checkpoint 的位置。
- CPU 测试、GPU peak memory 测试和 TorchTitan smoke 分别能证明什么。

## Patch 闭环

```bash
cat labs/l06_torchtitan_training/patch/task.md
$EDITOR labs/l06_torchtitan_training/patch/starter/selective_ckpt.py
make patch-test M=l06_torchtitan_training
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_output_matches_no_ckpt` | 输出与裸模型 allclose |
| `test_grads_match_no_ckpt` | 输入梯度和参数梯度与裸模型 allclose |
| `test_attention_only_policy_counts` | `attn0`、`attn1` 被包，`mlp0`、`mlp1` 未被包 |
| `test_all_false_policy_is_noop` | policy 全 False 时直接 child 类型不变 |
| `test_memory_drops_with_attn_ckpt` | GPU 可用时比较 checkpoint 前后 peak memory |

GPU 测试 skip 只能说明当前环境没有 CUDA 验证条件，不代表显存收益已被证明。

## Smoke 闭环

```bash
python labs/l06_torchtitan_training/scripts/run_torchtitan_stub.py --config configs/4090_debug.toml --mode smoke
```

这个 smoke 是 TorchTitan 边界教学 stub：它跑一个小训练循环，写 metrics、训练 checkpoint、resume 校验和 `framework_validation.json`。它验证训练框架 artifact 链路，不等价于真实 TorchTitan 集群训练。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查数值不等价、policy 误包、GPU memory skip、RNG 和 wrapper 副作用 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 selective checkpoint wrapper 与 TorchTitan `apply_ac` 主路径 |
| [outputs/training_step_template.md](outputs/training_step_template.md) | 记录一次 activation checkpoint 或 TorchTitan stub 复盘 |

## 进入下一讲

`make patch-test M=l06_torchtitan_training` 通过，并能解释 activation checkpoint 的显存和计算代价后，进入 [L08 Megatron 数据预处理](../l07_dataset_megatron_bin/README.md)。
