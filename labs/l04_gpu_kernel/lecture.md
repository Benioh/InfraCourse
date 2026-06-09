# L05 讲义：Triton 行 Softmax

## 0. 本讲目标

学完这一讲，你应该能完成五件事：

- 解释 softmax 为什么要先减行最大值再做 `exp`。
- 写出一个每个 Triton program 处理一行的 row softmax kernel。
- 正确使用 `BLOCK_SIZE`、`tl.arange`、mask load、`other=-inf` 和 mask store。
- 用测试区分 fp32 对齐、fp16 对齐、row sum、短行和长行边界。
- 用 roofline 语言说明 softmax 为什么常受 HBM 读写限制，并写清 benchmark 条件。

本讲不是完整 FlashAttention，也不实现 backward。它只抓住 FlashAttention、MoE router、采样 logits 和很多 fused kernel 都会复用的一条基础路径：对一行数做稳定 reduction，然后把结果写回 GPU。

## 1. 真实问题：softmax 公式简单，kernel 容易写错

softmax 的数学式很短：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

但是把它写成 GPU kernel 时，至少有四个工程问题：

1. `exp(x_i)` 可能溢出，尤其是 fp16。
2. 行长度不一定是 2 的幂，Triton program 的向量宽度却通常按 2 的幂选择。
3. padding 位置不能参与 max 和 sum，也不能写回越界地址。
4. softmax 算术量不大，但要读 logits、写 probabilities，还可能产生中间数据，性能经常受内存带宽影响。

本讲 patch 要求实现：

```python
def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """x: (n_rows, n_cols), CUDA tensor. Return row-wise softmax."""
```

输出要与 `torch.nn.functional.softmax(x, dim=-1)` 数值等价。这里的比较对象是 PyTorch 的 softmax；指标是 max absolute error 和 row sum；测试条件包括 dtype、shape 和是否有 CUDA/Triton。没有 CUDA 时测试会 skip，这不是通过证据。

## 2. 数值稳定 softmax

### 定义

稳定 softmax 用下面的形式计算：

```text
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
m = max_j x_j
```

### 直觉

同一行里所有元素都减去同一个常数，概率不会变。原因是分子和分母同时乘上 `exp(-m)`，这个因子会抵消。变化的是浮点路径：最大元素变成 0，其它元素小于等于 0，`exp` 的输入进入更安全的范围。

### 输入、中间状态、输出

输入是一行 logits。第一步求 `row_max`；第二步计算 `numer = exp(row - row_max)`；第三步求 `denom = sum(numer)`；第四步输出 `numer / denom`。

如果不减 max，大 logits 会让 `exp` 产生 `inf`。如果 `inf` 同时出现在分子和分母，结果可能变成 NaN。fp16 的可表示范围更窄，所以这个问题更容易出现。fp32 范围更大，但极端 logits 仍会溢出。

### 边界

减 max 只解决溢出风险，不解决 padding、越界或超长行分块问题。padding 位置如果被填成 0，`exp(0)=1` 会进入分母，导致概率变小。对 max reduction 来说，padding 的 identity 应该是 `-inf`。

## 3. Triton 的 program 模型

Triton kernel 不是直接写“一个线程处理一个元素”的 CUDA C 风格。本讲用一个更适合初学的模型：每个 Triton program 处理一整行。

Python wrapper 选择：

```text
grid = (n_rows,)
```

kernel 里：

```text
row_idx = tl.program_id(0)
col_offsets = tl.arange(0, BLOCK_SIZE)
mask = col_offsets < n_cols
```

`row_idx` 决定当前 program 处理哪一行。`col_offsets` 是这一行内的列偏移向量。`BLOCK_SIZE` 是编译期常量，表示这个 program 一次处理多少列。

当 `n_cols=300` 时，`BLOCK_SIZE = triton.next_power_of_2(300) = 512`。这样 reduction primitive 更容易走规则路径，但 300 到 511 的位置是 padding。mask 的作用是把真实列和 padding 列分开。

## 4. Mask Load 与 Mask Store

加载一行时，地址形如：

```text
input_ptr + row_idx * input_row_stride + col_offsets
```

真实列用 `tl.load` 读取。padding 列用 `other=-float("inf")`：

```python
row = tl.load(row_ptrs, mask=mask, other=-float("inf"))
```

这一步的输出是长度为 `BLOCK_SIZE` 的向量。真实列是 logits，padding 列是 `-inf`。

为什么是 `-inf`？因为后面先做 `tl.max(row, axis=0)`。`max(x, -inf)=x`，padding 不会变成虚假最大值。再往后 `exp(-inf)=0`，padding 也不会污染分母。

写回时也要带 mask：

```python
tl.store(output_ptrs, out, mask=mask)
```

只在 load 上 mask 不够。没有 store mask 时，padding 列可能写到当前行末尾之外，轻则结果错，重则触发 illegal memory access。

## 5. Wrapper 的责任

Triton kernel 负责行内计算，Python wrapper 负责发射前的边界检查和参数准备。

wrapper 的输入是一个 `torch.Tensor`。本关要求：

- 必须是 CUDA tensor。
- 必须是 2D。
- shape 是 `(n_rows, n_cols)`。
- `BLOCK_SIZE = triton.next_power_of_2(n_cols)`。
- 输出用 `torch.empty_like(x)` 分配。
- 启动 grid 为 `(n_rows,)`。

wrapper 不应该调用 `F.softmax` 或 `torch.softmax` 兜底。任务目标是写 Triton kernel；CPU 回退或 PyTorch 回退会绕过要学习的地址、mask 和 reduction 合同。

## 6. 测试如何证明最小合同

运行：

```bash
make patch-test M=l04_gpu_kernel
```

5 个测试覆盖不同维度：

| 测试 | 证明什么 |
|---|---|
| `test_matches_torch_softmax_fp32` | fp32 下与 PyTorch softmax 的 max diff 足够小 |
| `test_matches_torch_softmax_fp16` | fp16 下仍能在容忍度内对齐 |
| `test_row_sums_to_one` | 每行概率和接近 1 |
| `test_short_rows` | 短行没有特殊分支错误 |
| `test_long_rows` | 2048 列长行仍能在单 program 中处理 |

这些测试不覆盖 backward、不覆盖超长行分块、不覆盖 causal mask、不覆盖真实性能。没有 CUDA 时会自动 skip。报告中要把 skip 写成“未验证”，不能写成“通过”。

## 7. Memory-Bound 与 Roofline 直觉

softmax 每个元素要做 `exp`、加法、除法，但从系统角度看，它通常先按内存带宽分析。原因是它需要读 logits、写输出，并且朴素实现还可能把 max、exp、sum、division 拆成多个 kernel 或多个中间 tensor，增加 HBM 往返。

fused softmax 的收益来自减少中间读写和 kernel launch。它不改变 softmax 数学。MiniInfra 的 `roofline_softmax` 会估算：

- fused softmax 的读写字节数。
- eager softmax 的读写字节数。
- 给定 GPU profile 的 peak HBM bandwidth。
- 粗略 occupancy 和 estimated time。

这些估算只用于建立预算直觉。真实性能结论必须说明：

- 比较对象：例如 PyTorch `F.softmax`。
- shape：`n_rows`、`n_cols`。
- dtype：fp32、fp16 或 bf16。
- GPU 型号。
- 计时方式：warmup、重复次数、同步点。
- 指标：kernel time、bandwidth_gbs、max_abs_err、row_sum。
- 代价：实现复杂度、适配 shape 的范围、资源占用。

没有这些条件，只能说“预期瓶颈在 HBM IO”，不能写成 benchmark 结论。

## 8. Online Softmax 是什么

本讲 patch 的 Triton kernel 假设一行可以放进一个 program。长上下文 attention 里，一行可能很长，单个 block 放不下。这时需要 online softmax。

online softmax 把一行切成多个 chunk，维护两个状态：

```text
running_max
running_sum
```

读到新 chunk 时，先计算 `block_max`，再得到 `new_max = max(running_max, block_max)`。如果最大值变了，旧的 `running_sum` 必须乘上 `exp(running_max - new_max)` 重新缩放，然后再加上当前 chunk 的指数和。

```text
running_sum =
  running_sum * exp(running_max - new_max)
  + sum(exp(block - new_max))
```

这个重缩放是 online softmax 的关键。没有它，旧 chunk 的分母会用旧 max 标尺，新 chunk 用新 max 标尺，两个分母不能相加。FlashAttention 的长序列路径就在更复杂的 tile 调度里维护类似状态。

## 9. Debug 路线

结果不对时先分三类。

数值错误：

- 是否减了 row max。
- padding 是否填 `-inf`。
- row sum 是否接近 1。
- fp16/fp32 容忍度是否合理。

越界或 illegal memory access：

- `grid=(n_rows,)` 是否正确。
- `row_idx * stride + col_offsets` 是否正确。
- load 和 store 是否都带 mask。
- 用非 2 的幂行长复现，例如 `n_cols=300` 或 `n_cols=4097`。

资源或编译错误：

- `BLOCK_SIZE` 是否过大。
- 中间变量是否太多。
- `num_warps`、`num_stages` 是否需要调整。
- 当前 GPU 的 shared memory 和 register 资源是否足够。

每次只改一个变量。报告里保留命令、shape、dtype、GPU、Triton 版本、max diff、row sum 和错误信息。

## 10. 本讲小结

L05 把系统视角压到一个 row softmax kernel。核心链路是：wrapper 检查输入并按行启动 grid；每个 Triton program 用 mask 读取一行；padding 用 `-inf` 作为 max reduction 的 identity；行内减 max、exp、sum、divide；最后用 mask 写回真实列。patch-test 验证功能语义，bench 脚本帮助理解 online softmax 和 roofline 预算。进入后续并行课程前，你需要能同时解释“算对”和“跑得快”分别需要哪些证据。
