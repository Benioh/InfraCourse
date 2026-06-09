# L24 Patch · AWQ-lite W8 Per-Channel Quantization

## 你要交付什么

实现 AWQ 思路的简化版：**用 activation 统计指导 per-channel scale 选择**，再做对称 INT8 量化：

```python
def compute_awq_scale(weight: Tensor, act_amax: Tensor, alpha: float = 0.5) -> Tensor:
    """每输入通道一个 AWQ scale: s_i = act_amax_i^alpha / w_amax_i^alpha."""

def quantize_w8_per_channel(weight: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    """对 (weight * scale) 做 per-output-channel 对称 INT8 量化，
    返回 int_weight (int8) 和 per-channel scale (float)。"""

def dequantize_w8_per_channel(int_weight: Tensor, per_out_scale: Tensor, awq_scale: Tensor | None = None) -> Tensor:
    """把 int_weight + per_out_scale 还原；传入 awq_scale 时再除回原始 weight 空间。"""
```

**禁止** `bitsandbytes` / `auto-gptq` 等库。
**允许** torch 基础算子。

补丁规模目标：50–80 行。

## 算法

### Per-channel scale (AWQ 核心 idea)

激活里有 outlier（某些 channel 的值特别大），直接量化 weight 会让 outlier channel 的精度变差。
AWQ 的做法：把 outlier 部分"吸"到 weight 里。

```
W_out = W * s           # s 的 shape 是 (in_features,)
S = clamp(act_amax^α / w_amax_in^α, min=1.0)
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
2. activation outlier 越大，对应输入通道的 AWQ scale 应该越大。
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
| `test_per_channel_scale_shape` | AWQ scale shape = (in_features,) |
| `test_awq_scale_responds_to_outliers` | outlier activation 对应通道的 scale 均值更大 |
| `test_alpha_zero_equals_naive` | alpha=0 退化为纯 INT8 weight 量化 |

## 写完之后你能做什么

- 解释 AWQ / SmoothQuant / GPTQ 的核心 idea。
- 判断 W8A16 量化是否真的降低显存，并知道速度结论还需要真实 kernel benchmark。
- 看懂 vLLM / SGLang 的量化 weight 加载流程。
