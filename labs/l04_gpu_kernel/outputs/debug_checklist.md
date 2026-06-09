# L05 Triton Softmax Debug Checklist

这份清单用于排查 L05 的 row softmax kernel。先判定问题类型，再改代码；不要同时调整公式、mask、BLOCK_SIZE 和 dtype。

## 1. 先固定现场

- 命令：`make patch-test M=l04_gpu_kernel` 或具体 pytest case。
- 环境：GPU 型号、CUDA、PyTorch、Triton 版本。
- 输入：`n_rows`、`n_cols`、dtype、stride、是否 contiguous。
- 指标：max_abs_err、row sum、失败 case 名称、错误信息。
- 状态：pass、skip、failure 要分开记录。

## 2. 数值错误

| 现象 | 优先检查 |
|---|---|
| max diff 很大 | 是否减了 `row_max`，是否漏掉 denominator |
| 出现 NaN/inf | 是否直接 `exp(row)`，padding 是否污染了 denominator |
| row sum 不接近 1 | `tl.sum(numer)` 是否按行内向量求和，store 是否写错地址 |
| fp32 过、fp16 不过 | 是否在低精度下溢出/溢出，容忍度是否按测试要求 |

对应源码：

- `labs/l04_gpu_kernel/patch/reference/triton_softmax.py` L34-L37
- `mini_infra/gpu/triton_softmax.py` L31-L40

## 3. Mask 和越界

- `mask = col_offsets < n_cols` 是否正确。
- `tl.load(..., mask=mask, other=-float("inf"))` 是否使用 `-inf`。
- `tl.store(..., mask=mask)` 是否存在。
- 地址是否使用 `row_idx * row_stride + col_offsets`。
- 用非 2 的幂行长复现，例如 `n_cols=300`。

快速判断：

```text
load mask 保护读越界。
other=-inf 保护 max/sum 语义。
store mask 保护写越界。
```

对应源码：

- `labs/l04_gpu_kernel/patch/reference/triton_softmax.py` L26-L33
- `labs/l04_gpu_kernel/patch/reference/triton_softmax.py` L38-L42

## 4. Wrapper 和 Launch

- 是否拒绝 CPU tensor。
- 是否要求 `x.dim() == 2`。
- `n_rows, n_cols = x.shape` 是否正确。
- `BLOCK_SIZE = triton.next_power_of_2(n_cols)` 是否正确。
- output 是否 `torch.empty_like(x)`。
- kernel launch 是否使用 `softmax_kernel[(n_rows,)](...)`。
- stride 是否传入 `x.stride(0)` 和 `output.stride(0)`。

对应源码：

- `labs/l04_gpu_kernel/patch/reference/triton_softmax.py` L45-L60

## 5. 测试状态

| 状态 | 解释 | 报告写法 |
|---|---|---|
| pass | GPU/Triton 环境中测试实际运行并通过 | 写明 shape、dtype、case |
| skip | 没有 CUDA，测试未运行 kernel | 写成未验证 |
| failure | kernel 运行或数值断言失败 | 附错误信息和定位路径 |

不要把 `bench_softmax.py` 的 JSON 输出当作 patch-test 通过。bench 是 MiniInfra CPU 模拟。

## 6. 性能排查

性能结论必须先固定：

- 比较对象：PyTorch `F.softmax`、reference kernel 或其他实现。
- shape：`n_rows`、`n_cols`。
- dtype：fp32/fp16/bf16。
- 硬件：GPU 型号。
- 计时：warmup、重复次数、同步点。
- 指标：kernel time、bandwidth_gbs、max_abs_err、row_sum。

常见方向：

- bandwidth 低：看 stride 是否连续、BLOCK_SIZE 是否合适、launch overhead 是否主导。
- 编译或资源错误：看 BLOCK_SIZE、num_warps、num_stages 和中间变量数量。
- shape 变化后变慢：重新扫 BLOCK_SIZE/warps，不要把一个硬件配置直接搬到所有 shape。
