# L03 · TorchTitan：Selective Activation Checkpoint

> 本关只做一件事：**实现一个 selective activation checkpoint wrapper**——让你能"只 ckpt attention 不 ckpt MLP"。

写完这关你能解释 TorchTitan / Megatron 的 ckpt policy 接口。

## 闭环

```bash
cat labs/l06_torchtitan_training/patch/task.md
$EDITOR labs/l06_torchtitan_training/patch/starter/selective_ckpt.py
make patch-test M=l06_torchtitan_training   # 5 个测试，CPU 4 个 + GPU 1 个（无 GPU 自动 skip）
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_output_matches_no_ckpt` | 输出与裸模型 allclose |
| `test_grads_match_no_ckpt` | 梯度与裸模型 allclose |
| `test_attention_only_policy_counts` | attention 层被包，MLP 层未被包 |
| `test_all_false_policy_is_noop` | 全 False policy 不修改模型 |
| `test_memory_drops_with_attn_ckpt` (gpu) | 显存峰值下降 |

## 卡住怎么办

1. 看 `notebooks/n06_activation_recompute.ipynb`。
2. `make patch-hint M=l06_torchtitan_training`。
3. `make patch-show-solution M=l06_torchtitan_training`。

## 进入下一关

`make patch-test M=l06_torchtitan_training` 全绿后，继续做源码理解口试。下一关 [L04 Megatron 预训练](../l08_megatron_text_pretrain/README.md) 让你给 Megatron 加一个新的 LR scheduler。
