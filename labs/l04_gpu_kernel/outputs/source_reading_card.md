# L05 Source Reading Card

## 一句话主线

`triton_softmax(x)` 为每一行启动一个 Triton program，mask 读取 logits，padding 填 `-inf`，行内减 max 后计算 `exp/sum/divide`，最后只把真实列写回。

## 必读源码

| 顺序 | 文件 | 行号 | 记住什么 |
|---|---|---|---|
| 1 | `labs/l04_gpu_kernel/patch/starter/triton_softmax.py` | L27-L49 | kernel TODO：row id、offset、mask、load、max、exp、sum、store |
| 2 | `labs/l04_gpu_kernel/patch/starter/triton_softmax.py` | L52-L70 | wrapper TODO：CUDA/2D 检查、BLOCK_SIZE、output、launch |
| 3 | `labs/l04_gpu_kernel/patch/reference/triton_softmax.py` | L26-L42 | reference kernel 的 masked load、稳定 softmax 和 masked store |
| 4 | `labs/l04_gpu_kernel/patch/reference/triton_softmax.py` | L45-L60 | reference wrapper 的 shape、block 和 grid |
| 5 | `labs/l04_gpu_kernel/patch/tests/test_patch.py` | L31-L77 | fp32、fp16、row sum、short row、long row |
| 6 | `mini_infra/gpu/triton_softmax.py` | L31-L63 | stable softmax 和 online softmax 数学 |
| 7 | `mini_infra/gpu/memory_model.py` | L83-L132 | bytes、occupancy、roofline softmax 估算 |
| 8 | `labs/l04_gpu_kernel/scripts/bench_softmax.py` | L12-L23 | CPU 侧教学模拟入口 |

## 关键判断

| 问题 | 判断方式 |
|---|---|
| 是否数值稳定 | 看是否 `row - row_max` 后再 `tl.exp` |
| padding 是否污染结果 | 看 load 的 `other=-inf` 和 store 的 mask |
| BLOCK_SIZE 是否覆盖行 | 看 `next_power_of_2(n_cols)` 和 `col_offsets < n_cols` |
| 是否真的跑 kernel | 看 GPU patch-test 是否 pass，而不是 skip |
| bench 能证明什么 | 只证明 CPU 模拟和 roofline 估算链路 |

## 复述模板

```text
本讲 kernel 每个 program 处理一行。
wrapper 根据 n_cols 选择 2 的幂 BLOCK_SIZE，并按 n_rows 启动 grid。
kernel 用 mask 读取真实列，padding 用 -inf。
row_max 让 exp 输入不超过 0。
numer / denom 得到概率。
store 时复用 mask，避免写越界。
```

## 常见误读

- `other=0` 不安全；softmax 的 max reduction 需要 `-inf`。
- 只要 row sum 为 1 就一定正确；还要和 PyTorch softmax 比 max diff。
- skip 不等于 pass；没有 CUDA 时 kernel 没有运行。
- roofline 估算不等于真实 benchmark；真实结论需要 GPU timing。
