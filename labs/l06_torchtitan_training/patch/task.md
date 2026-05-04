# L03 Patch · Selective Activation Checkpoint

## 你要交付什么

实现一个**选择性激活检查点**（selective activation checkpoint）：让你只对模型中的某些层（比如 attention）应用 `torch.utils.checkpoint`，其他层保留 activation。

这是 TorchTitan / Megatron 的 ckpt policy 在做的事情——不是 "全 ckpt 或全不 ckpt"，而是按层挑。

```python
def selective_checkpoint_wrap(model, policy_fn) -> nn.Module:
    """对 model.named_children() 中 policy_fn(name, child) 返回 True 的子模块，
    用 torch.utils.checkpoint 包起来。返回原模型（in-place 修改）。"""

def attention_only_policy(name: str, module: nn.Module) -> bool:
    """policy 函数：只对 'attn' 或 'attention' 的 child 返回 True。"""
```

**禁止** 用 `fairscale` / `deepspeed` 现成 API。
**允许** `torch.utils.checkpoint` / `nn.Module` 全部内置 API。

补丁规模目标：30–60 行。

## 接口契约

```python
model = build_transformer()  # Sequential of [embed, attn1, mlp1, attn2, mlp2, head]
selective_checkpoint_wrap(model, attention_only_policy)
# 现在 model 的 attn1/attn2 在 forward 时会自动 checkpoint，
# embed/mlp1/mlp2/head 保持正常（保留 activation）

x = torch.randn(...)
y = model(x)
y.sum().backward()  # forward 跑一次，attn 部分会再跑一次
```

## 不变量

1. 任何 policy 下，输出与未包 ckpt 的等价（数值 atol=1e-5）。
2. 梯度也等价。
3. policy 全 False 时，模型未被改动（`isinstance(child, ...)` 保持）。
4. policy 全 True 时，所有 child 都被 ckpt。
5. wrap 之后 `model.parameters()` 还是同一组（不复制权重）。

## 怎么验证

```bash
make patch-test M=l06_torchtitan_training
```

5 个测试，CPU 友好（peak memory 测试有 GPU 跑、无则 skip）：

| 测试 | 验证 |
|---|---|
| `test_output_matches_no_ckpt` | 输出与裸模型 allclose |
| `test_grads_match_no_ckpt` | grads 与裸模型 allclose |
| `test_attention_only_policy_counts` | 在固定模型上 policy 包了正确数量的层 |
| `test_all_false_policy_is_noop` | policy 全 False 时模型未变 |
| `test_memory_drops_with_attn_ckpt` (gpu) | GPU 上峰值显存 ckpt 后下降 |

## 写完之后你能做什么

- 看懂 TorchTitan `parallelize_llama.py` 里 `apply_ac` 的 policy 接口。
- 在 Capstone 给多模态模型挑哪些层 ckpt（image/audio encoder 不 ckpt 因为它们冻结）。
- 解释 "ckpt 是用算力换显存"——recompute 的代价就是多一次 forward。
