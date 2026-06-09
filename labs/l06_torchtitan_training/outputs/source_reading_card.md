# L07 Source Reading Card

## 一句话主线

`selective_checkpoint_wrap` 遍历模型的直接 child，policy 返回 True 时用 `_CheckpointWrapper` 原地替换；wrapper 持有原 child，并在 forward 中调用 `torch.utils.checkpoint`，让 backward 重算被包模块。

## 必读源码

| 顺序 | 文件 | 行号 | 记住什么 |
|---|---|---|---|
| 1 | `patch/starter/selective_ckpt.py` | L24-L33 | wrapper 保存原 module，forward 调 checkpoint |
| 2 | `patch/starter/selective_ckpt.py` | L36-L55 | selective wrap 和 attention policy 的 TODO |
| 3 | `patch/reference/selective_ckpt.py` | L13-L31 | 最小正确实现 |
| 4 | `patch/tests/test_patch.py` | L65-L123 | 输出、梯度、policy count、no-op |
| 5 | `patch/tests/test_patch.py` | L126-L148 | GPU peak memory 测试，CUDA 缺失时 skip |
| 6 | `torchtitan/distributed/activation_checkpoint.py` | L204-L255 | TorchTitan `apply_ac` 主路径 |
| 7 | `torchtitan/models/llama3/parallelize.py` | L80-L102 | AC 在 compile/FSDP 前后的位置 |
| 8 | `scripts/run_torchtitan_stub.py` | L50-L105 | 训练循环、metrics、磁盘 checkpoint 和 framework validation |

## 关键判断

| 问题 | 判断方式 |
|---|---|
| 是否复制参数 | wrapper 内应保存原 child 到 `self.module` |
| policy 是否正确 | `attn0/attn1` 被包，`mlp0/mlp1` 未被包 |
| 输出是否等价 | wrapped 输出与 base 输出 allclose |
| 梯度是否等价 | input grad 和 param grad allclose |
| 显存是否证明下降 | CUDA test 实际运行且 wrap_peak < base_peak |

## 复述模板

```text
activation checkpoint 用额外 forward 计算换 activation 显存。
policy 决定哪些直接 child 被包。
wrapper 不复制参数，只改变被包模块的 forward 保存策略。
forward 输出应与裸模型一致。
backward 重算被包模块，梯度也应一致。
```

## 常见误读

- activation checkpoint 与磁盘训练 checkpoint 是两个概念。
- checkpoint 不减少参数、梯度或 optimizer state。
- GPU memory test skip 不能写成显存下降。
- policy 全 False 时应保持直接 child 类型不变。
