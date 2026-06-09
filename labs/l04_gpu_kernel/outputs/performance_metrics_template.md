# L05 Performance Metrics Template

## 1. Run 信息

- 日期：
- git commit：
- GPU：
- CUDA / driver：
- PyTorch：
- Triton：
- 命令：
- case：patch-test / bench / smoke / benchmark

## 2. 输入配置

| 字段 | 值 |
|---|---|
| `n_rows` |  |
| `n_cols` |  |
| dtype |  |
| contiguous |  |
| `BLOCK_SIZE` |  |
| num_warps |  |
| num_stages |  |

## 3. 功能正确性

| 指标 | 值 | 判断 |
|---|---|---|
| fp32 max_abs_err |  |  |
| fp16 max_abs_err |  |  |
| max row_sum_err |  |  |
| NaN/inf count |  |  |
| short_rows result |  | pass / skip / fail |
| long_rows result |  | pass / skip / fail |

## 4. 性能记录

| 比较对象 | kernel time | bandwidth_gbs | 条件 |
|---|---:|---:|---|
| Triton starter |  |  |  |
| PyTorch F.softmax |  |  |  |
| MiniInfra roofline estimate |  |  | CPU 模拟，不是实测 |

计时方法：

- warmup：
- repeats：
- 是否 `torch.cuda.synchronize()`：
- profiler / do_bench / ncu：

## 5. 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| masked load | `patch/reference/triton_softmax.py` L29-L33 |  |
| stable math | `patch/reference/triton_softmax.py` L34-L37 |  |
| masked store | `patch/reference/triton_softmax.py` L38-L42 |  |
| wrapper launch | `patch/reference/triton_softmax.py` L51-L59 |  |
| roofline bytes | `mini_infra/gpu/memory_model.py` L83-L92 |  |

## 6. 结论

- 本次能证明：
- 本次不能证明：
- 如果失败，最小复现命令：
- 下一步要改的变量：
