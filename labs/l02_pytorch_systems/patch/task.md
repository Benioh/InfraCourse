# L01 Patch · 显存账本（Memory Accounting）

## 你要交付什么

在 `starter/memory_probe.py` 里实现 3 个函数，把模型的显存拆成可解释的几项：

```python
def count_param_bytes(model: nn.Module) -> int: ...
def count_grad_bytes(model: nn.Module) -> int: ...
def count_optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int: ...
```

写完之后你能对任意模型回答："params 占多少、grads 占多少、optimizer states 占多少？"——这是面试官最常问、也最能区分新手与老手的 infra 问题。

**禁止使用** 第三方库（torchsummary、torchinfo 等）。
**允许使用** `torch.Tensor.numel()`、`element_size()`、`torch.optim.Optimizer.state` 等内置 API。

补丁规模目标：30–60 行。

## 接口契约

```python
def count_param_bytes(model: nn.Module) -> int:
    """所有 nn.Parameter 占用的字节数总和（fp32 一个 param ≈ 4 * numel）"""

def count_grad_bytes(model: nn.Module) -> int:
    """所有 .grad 占用的字节数总和。
    backward 之后 grad 存在；optimizer.zero_grad(set_to_none=True) 后 grad=None，应返回 0。"""

def count_optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """所有 optimizer.state[p] 中 Tensor 类型 entry 的字节数总和。
    plain SGD = 0；SGD+momentum ≈ 1×param_bytes；Adam ≈ 2×param_bytes（m + v）。"""
```

## 不变量

1. `count_param_bytes(model)` 与 `sum(p.numel() * p.element_size() for p in model.parameters())` 完全相等。
2. backward 之后，`count_grad_bytes(model) == count_param_bytes(model)`（fp32 grads 与 params 同形）。
3. `optimizer.zero_grad(set_to_none=True)` 之后，`count_grad_bytes(model) == 0`。
4. plain SGD 一步之后 `count_optimizer_state_bytes(opt) == 0`；Adam 一步之后 `>= 2 * param_bytes`。

## 怎么验证

```bash
make patch-test M=l02_pytorch_systems
```

7 个测试，全是 CPU 友好的结果对比：

| 测试 | 验证 |
|---|---|
| `test_count_param_bytes` | 与手算公式完全相等 |
| `test_count_grad_bytes_after_backward` | grad_bytes 与 param_bytes 相等 |
| `test_count_grad_bytes_zero_when_none` | grad=None 时返回 0 |
| `test_optimizer_state_sgd_zero` | plain SGD 状态为 0 |
| `test_optimizer_state_sgd_momentum` | SGD+momentum 状态 ≈ 1×param_bytes |
| `test_optimizer_state_adam_2x_params` | Adam 状态 ≈ 2×param_bytes |
| `test_works_with_mixed_dtype` | half + float 混合模型字节计算正确 |

## 写完之后你能做什么

- 面试遇到"7B 模型用 Adam 训练需要多少显存？" 你能立刻拆分出 params + grads + optimizer + activations 四项。
- L05 Megatron 优化、L05.5 MoE、L11 SLiME 权重同步都会复用这套账本。
- Capstone Stage A 训练 projector 时直接用这套函数算显存预算。
