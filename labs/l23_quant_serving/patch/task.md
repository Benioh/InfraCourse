# L08.5 Patch · AWQ-lite Per-Channel Quantization

## 你要交付什么

实现 AWQ 思路的简化版：**用 activation 统计指导 per-channel scale 选择**，再做对称 INT8 量化：

```python
def compute_awq_scale(weight: Tensor, act_amax: Tensor, alpha: float = 0.5) -> Tensor:
    """每输出通道一个 scale: s_c = act_amax^alpha / w_amax_c^alpha. 让 outlier act 被 absorb 进 weight."""

def quantize_w8_per_channel(weight: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    """对 (weight * scale) 做 per-output-channel 对称 INT8 量化，
    返回 int_weight (int8) 和 per-channel scale (float)。"""

def dequantize_w8_per_channel(int_weight: Tensor, scale: Tensor) -> Tensor:
    """把 int_weight + scale 还原为 fp32 weight（应该 ≈ 原 weight）。"""
```

**禁止** `bitsandbytes` / `auto-gptq` 等库。
**允许** torch 基础算子。

补丁规模目标：50–80 行。

## 算法

### Per-channel scale (AWQ 核心 idea)

激活里有 outlier（某些 channel 的值特别大），直接量化 weight 会让 outlier channel 的精度变差。
AWQ 的做法：把 outlier 部分"吸"到 weight 里。

```
W_out = W * s           # 数学等价：x @ W = (x / s) @ (W * s)
S = (act_amax^α / w_amax_c^α) ** clip[1.0, ...]
```

α=0.5 是经验值。极端情况：
- α=0：纯 weight 量化（no AWQ）
- α=1：weight 完全适应 activation amax

### 对称 INT8 量化

```
scale_c = max(|W_out_c|) / 127        # 每输出通道一个 scale
int_w = round(W_out / scale_c).clip(-127, 127).int8()
dequantized = int_w * scale_c         # ≈ W_out
```

## 不变量

1. dequantize(quantize(W)) 与 W 的相对误差 < 5% (fp32 → int8 一般 ~2-3%)。
2. activation outlier 越大，AWQ scale (alpha>0) 比纯 INT8 (alpha=0) 误差越小。
3. int_weight 严格在 [-127, 127] 范围内。

## 怎么验证

```bash
make patch-test M=l23_quant_serving
```

5 个测试：

| 测试 | 验证 |
|---|---|
| `test_int8_range` | int_weight 全在 [-127, 127] |
| `test_quantize_dequantize_roundtrip` | 误差 < 5% |
| `test_per_channel_scale_shape` | scale shape = (out_features,) |
| `test_awq_better_with_outliers` | outlier act 时 alpha=0.5 比 alpha=0 误差小 |
| `test_alpha_zero_equals_naive` | alpha=0 退化为纯 INT8 weight 量化 |

## 写完之后你能做什么

- 解释 AWQ / SmoothQuant / GPTQ 的核心 idea。
- 给 Capstone 推理服务上 W8A16 加速（约 1.5-2× 推理快）。
- 看懂 vLLM / SGLang 的量化 weight 加载流程。
