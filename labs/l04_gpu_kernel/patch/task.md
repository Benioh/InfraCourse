# L01.7 Patch · Triton 行 Softmax

## 你要交付什么

在 `starter/triton_softmax.py` 里写一个 **数值稳定** 的行内 softmax Triton kernel，
其结果与 `F.softmax(x, dim=-1)` 在 atol=1e-5(fp32) / atol=1e-2(fp16) 下完全一致。

```python
def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """x: (n_rows, n_cols), CUDA tensor. 返回同形 softmax，沿最后一维归一。"""
```

**禁止** 调用 `F.softmax` / `torch.softmax` 偷懒。
**允许** 用 `triton`、`triton.language as tl` 全部 API。

补丁规模目标：30–60 行。

## 接口契约

```python
x = torch.randn(8, 256, device="cuda")
y = triton_softmax(x)
assert torch.allclose(y, F.softmax(x, dim=-1), atol=1e-5)
```

要点：
- 输入 2D `(n_rows, n_cols)`，按 row 启动 grid（`grid=(n_rows,)`）
- 用 `triton.next_power_of_2(n_cols)` 选 `BLOCK_SIZE`，用 mask 处理超出部分
- **必须** 减最大值再 exp（数值稳定 / online softmax）
- 支持 fp32 和 fp16（fp16 时减最大值更关键）

## 不变量

1. 输出与 `F.softmax(x, dim=-1)` 数值等价。
2. 每行输出之和 ≈ 1.0（softmax 定义）。
3. 输入全 0 时输出全 1/n。
4. 处理 `n_cols ∈ [16, 4096]` 的范围，更长可拒绝（assert / raise）。

## 怎么验证

```bash
make patch-test M=l04_gpu_kernel
```

5 个测试，**没 CUDA 自动跳过**：

| 测试 | 验证 |
|---|---|
| `test_matches_torch_softmax_fp32` | fp32 atol=1e-5 |
| `test_matches_torch_softmax_fp16` | fp16 atol=1e-2 |
| `test_row_sums_to_one` | 每行 sum ≈ 1 |
| `test_short_rows` | n_cols=64 |
| `test_long_rows` | n_cols=2048 |

## 写完之后你能做什么

- 解释 FlashAttention 的 online softmax 数学（你刚写过简化版）。
- 在 L08.7 spec decode、L05.5 MoE router 用同样 pattern 写小 kernel。
- 看懂 SGLang / Megatron 的自定义 Triton 内核（attention、layernorm、rmsnorm）。
