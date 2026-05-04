# L08.5 · 量化 Serving：AWQ-lite Per-Channel

> 本关只做一件事：**实现 AWQ 思路的 W8 per-channel 量化**——含 activation-aware scale。

## 闭环

```bash
cat labs/l23_quant_serving/patch/task.md
$EDITOR labs/l23_quant_serving/patch/starter/awq_calibrate.py
make patch-test M=l23_quant_serving
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_int8_range` | 量化值在 [-127, 127] |
| `test_quantize_dequantize_roundtrip` | 相对误差 < 5% |
| `test_per_channel_scale_shape` | scale shape = (in_features,) |
| `test_awq_better_with_outliers` | activation 有 outlier 时 alpha=0.5 不比 alpha=0 差 |
| `test_alpha_zero_equals_naive` | alpha=0 退化为纯 INT8 量化 |

## 卡住怎么办

1. 看 `notebooks/n17_quant_calibration.ipynb`。
2. `make patch-hint M=l23_quant_serving`。
3. `make patch-show-solution M=l23_quant_serving`。

## 进入下一关

`make patch-test` 全绿后，继续做源码理解口试。下一关 [L08.7 Spec Decode](../l24_spec_decode/README.md)。
