# L06 讲义：手写 Tensor Parallel Linear

## 0. 本讲目标

学完这一讲，你应该能完成六件事：

- 区分 Tensor Parallelism（TP）和 Data Parallelism（DP）解决的系统压力。
- 根据 `nn.Linear` 的 weight shape 判断 Column/Row 的切分维度。
- 写出 `_CopyToParallelRegion`、`_ReduceFromParallelRegion`、`_GatherAlongLastDim` 的双向规则。
- 实现 `ColumnParallelLinear` 和 `RowParallelLinear` 的参数分片、forward 和 bias 处理。
- 用 2-rank CPU/gloo 测试证明输出和梯度对齐单卡 `nn.Linear`。
- 说明 TP 的通信、性能和 checkpoint 边界。

## 1. 真实问题：一个 Linear 太大时，DP 不够

数据并行让每个 rank 持有完整模型，处理不同 batch，然后同步梯度。它能增加样本吞吐，但每个 rank 仍然保存完整 `Linear` 权重、梯度和 optimizer state。

当一个层内矩阵太大时，单纯 DP 不能降低单 rank 的层内参数压力。Transformer 里的 MLP、attention projection、MoE expert FFN 和多模态 projector 都可能遇到这种问题。TP 的做法是把同一个层拆给多个 rank 共同计算。

本讲只处理 `nn.Linear`：

```text
y = x @ W.T + b
W shape = (out_features, in_features)
```

这行 shape 是后面所有切分的锚点。PyTorch 的 `weight` 采用 `(out, in)`，很多初学者会误记成 `(in, out)`。记错这一点，Column 和 Row 会直接切反。

## 2. Column Parallel：切输出维度

Column Parallel 把 `W(out, in)` 沿输出维度切开。对于 `world_size = 2`：

```text
W0: [out/2, in]
W1: [out/2, in]
b0: [out/2]
b1: [out/2]
```

每个 rank 都拿完整输入 `x[..., in]`，只计算自己负责的输出片段：

```text
out_local = x @ W_local.T + b_local
```

如果 `gather_output=True`，所有 rank 用 all-gather 把输出片段拼成完整 `[..., out]`。如果 `gather_output=False`，每个 rank 只返回自己的输出片段。

Column forward 不需要 all-reduce，因为不同 rank 计算的是不同输出坐标，不能求和。Column backward 的输入梯度需要求和：完整输入 `x` 被每个 rank 的 `W_local` 使用，链式法则要求把每个分片贡献的 `grad_x_local` 加起来。

所以本关用 `_CopyToParallelRegion` 表达这条规则：

```text
forward:  identity
backward: all_reduce(SUM) on grad_output
```

## 3. Row Parallel：切输入维度

Row Parallel 把 `W(out, in)` 沿输入维度切开。对于 `world_size = 2`：

```text
W0: [out, in/2]
W1: [out, in/2]
b : [out]  replicated
```

如果输入还没分片，先把 `x[..., in]` 沿最后一维切开，当前 rank 取自己的 `x_local[..., in/2]`。本地计算：

```text
out_local = x_local @ W_local.T
```

这里 `out_local` 已经是完整输出维度 `[..., out]`，但它只包含一部分输入维度的贡献。真正输出是所有 rank 的贡献和：

```text
out = all_reduce_sum(out_local)
```

Row bias 是最常见的错误点。bias 在每个 rank 上复制，但只能在 all-reduce 后加一次。如果把 bias 传给本地 `F.linear(x_local, W_local, bias)`，每个 rank 的 partial output 都带一份 bias，all-reduce 后会变成 `bias * world_size`。

所以 Row 的正确顺序是：

```python
out_local = F.linear(x_local, self.weight)
out = _ReduceFromParallelRegion.apply(out_local, self.process_group)
if self.bias is not None:
    out = out + self.bias
```

## 4. 三个 Autograd Primitive

分布式层不能只写 forward collective。backward 的梯度路径同样要定义清楚。starter 里有三个 `autograd.Function`。

### `_CopyToParallelRegion`

输入是一个 tensor 和 process group。forward 直接返回输入。backward 对 `grad_output` 做 all-reduce sum，再返回。

它用于 Column 输入路径。forward 时每个 rank 都看到完整输入；backward 时每个 rank 都产生一份输入梯度贡献，所以要求和。

### `_ReduceFromParallelRegion`

forward 对输入做 all-reduce sum。backward 直接返回 `grad_output`。

它用于 Row output 路径。forward 时每个 rank 产生 partial output，需要求和；backward 时上游完整输出梯度已经对每个 rank 可用，本关不再做 collective。

### `_GatherAlongLastDim`

forward all-gather 每个 rank 的最后一维分片，并沿最后一维 `cat`。backward 把上游梯度沿最后一维 chunk 成 `world_size` 份，当前 rank 取自己的那份。

它用于 Column 的 `gather_output=True`。gather 的输出是拼接结果，反向时每个输出片段只归属于对应 rank。

## 5. Patch Tests 怎么证明语义

运行：

```bash
make patch-test M=l05_distributed_primitives
```

测试 harness 用 multiprocessing 启动 2 个 worker，初始化 gloo process group，再运行 worker case。每个 worker 会构造 canonical 单卡 `nn.Linear`，把权重广播到所有 rank，再把对应切片复制到 TP layer。

5 个测试分别回答：

| 测试 | 证明 |
|---|---|
| Column forward | all-gather 后输出等于单卡 |
| Row forward | all-reduce 后输出等于单卡 |
| Column backward | `grad_x` 和本地 `grad_W` 切片等于单卡 |
| Row backward | 输入梯度切片和本地 `grad_W` 等于单卡 |
| Row bias | bias 没有在 reduce 前被重复加 |

这些测试用 CPU/gloo，证明的是数学语义和 autograd 路径。它们不证明 GPU/NCCL 性能，不覆盖 sequence parallel、async dgrad、checkpoint 转换或 fused kernel。

## 6. Toy 脚本和 smoke 的边界

`toy_column_parallel_linear.py` 用普通 PyTorch 张量演示 Column 切输出维度后再 `cat`，它只覆盖 forward 数学。

`collectives_demo.py` 展示 all-reduce、all-gather 和 broadcast 的输出字段，帮助你复习 collective 语义。

`run_lab.py` 会运行 collective demo、DDP toy train、Column TP toy 和 pipeline toy，并写出 metrics 和 report。它证明课程 artifact 链路可用，但不验收 starter 的 `ColumnParallelLinear` 和 `RowParallelLinear`。

因此，L06 的功能验收仍然是 `make patch-test M=l05_distributed_primitives`。

## 7. 性能和通信边界

TP 的收益来自把单 rank 的权重、梯度和 matmul 工作拆小。代价是 collective 进入关键路径：

- Column gather output：forward 可能 all-gather。
- Column input grad：backward 需要 all-reduce。
- Row partial output：forward 需要 all-reduce。
- Row backward：本关 reduce primitive 的 backward 是 identity。

是否更快取决于矩阵大小、batch/sequence、world size、互联带宽、通信是否能和计算重叠、下游层是否能直接消费分片输出。小矩阵、慢互联或频繁 gather 成完整 tensor 时，通信可能盖过计算收益。

写性能结论必须列出比较对象、硬件、backend、shape、dtype、world size、计时方法和误差指标。CPU/gloo patch-test 通过后，只能讨论语义正确性。

## 8. Checkpoint 边界

TP checkpoint 保存的是分片权重。TP=2 的 Column 分片 shape 是 `(out/2, in)`；TP=4 需要 `(out/4, in)`。直接把 TP=2 的本地切片加载到 TP=4 rank 上，shape 和语义都不对。

改变 TP size 时，通常需要：

```text
gather old shards -> reconstruct full weight -> split for new TP size
```

Row 同理，只是切分维度是输入维度。真实 Megatron 还要处理 bias、optimizer state、sequence parallel 和 sharded state dict。L06 只要求你知道“本地分片不是完整权重”，不能把 checkpoint 当成普通单卡 state dict。

## 9. Debug 路线

先按失败类型分流：

- 初始化失败：检查 `MASTER_ADDR`、`MASTER_PORT`、rank、world size。
- hang：确认所有 rank 进入同一个 collective，次数和顺序一致。
- forward diff：检查切分维度、gather/reduce 位置、bias。
- backward diff：检查 autograd primitive 的 backward。
- Row bias diff：确认 bias 没有传入本地 `F.linear`。

最小证据要包括：失败测试名、rank、max diff、tensor shape、dtype、每个 rank 最后一条日志。一次只改一个变量，复跑同一个 case。

## 10. 本讲小结

L06 的核心是把 `Linear` 拆成两种可组合 primitive。Column 切输出维度，forward 产生输出片段，backward 对输入梯度求和；Row 切输入维度，forward 对 partial output 求和，bias 在 reduce 后加一次。三个 autograd primitive 把 forward 和 backward 的通信规则固定下来。patch-test 用 2-rank CPU/gloo 验证语义，后续课程再把这些 primitive 放进 TorchTitan、Megatron 和真实 GPU/NCCL 性能路径。

---

## 补充 A：五种集合通信原语全景

前面的 lab（L03 讲了 all_reduce，本讲用了 all_gather）只覆盖了部分原语。这里系统对比所有五种，建立完整直觉。

### A.1 原语对比表

| 原语 | 输入 | 输出 | 通信量 (ring) | 典型用途 |
|---|---|---|---|---|
| **all_reduce** | 每个 rank 有 N 大小的 tensor | 每个 rank 得到聚合后的 N 大小 tensor（如 sum） | 2N(ws-1)/ws | DDP 梯度聚合 |
| **all_gather** | 每个 rank 有 N/ws 大小的 tensor | 每个 rank 得到完整 N 大小 tensor（拼接） | N(ws-1)/ws | FSDP 恢复完整参数、TP Column gather |
| **reduce_scatter** | 每个 rank 有 N 大小的 tensor | 每个 rank 得到 N/ws 大小的聚合结果 | N(ws-1)/ws | FSDP 梯度分片、ZeRO-2/3 |
| **broadcast** | 1 个 rank 有数据 | 所有 rank 得到相同数据 | N | 参数初始化广播、weight sync |
| **all_to_all** | 每个 rank 有 N 大小数据（分成 ws 块） | 每个 rank 收到来自每个 rank 的一块 | N(ws-1)/ws | MoE token routing (EP) |

### A.2 all_reduce = reduce_scatter + all_gather

一个重要的等价关系：

```
all_reduce(tensor) ≡ all_gather(reduce_scatter(tensor))
```

DDP 用 all_reduce 聚合梯度：每个 rank 最终都有完整的聚合梯度。
ZeRO-2/FSDP 用 reduce_scatter：每个 rank 只保留自己负责的那一段聚合梯度，省显存。

### A.3 reduce_scatter 详解

```python
# 4 个 rank，每个有完整梯度 [g0, g1, g2, g3]（每段代表一部分参数的梯度）
# reduce_scatter 后：
# rank 0 得到 sum(所有 rank 的 g0 段)
# rank 1 得到 sum(所有 rank 的 g1 段)
# rank 2 得到 sum(所有 rank 的 g2 段)
# rank 3 得到 sum(所有 rank 的 g3 段)
```

**为什么 FSDP/ZeRO 用 reduce_scatter 而不是 all_reduce？**

all_reduce 后每个 rank 都有完整聚合梯度 → 每个 rank 都要存完整梯度 → 显存大。
reduce_scatter 后每个 rank 只有 1/ws 的聚合梯度 → 只对自己负责的参数做 optimizer step → 显存省 ws 倍。

代价：optimizer step 后需要 all_gather 恢复完整参数给 forward 用。

### A.4 broadcast 详解

```python
# rank 0 初始化了模型参数
# broadcast 把 rank 0 的参数发给所有 rank
torch.distributed.broadcast(tensor, src=0)
```

用途：
- 模型初始化后广播参数（确保所有 rank 起点一致）
- RL 中 weight sync：train worker → inference worker
- 某些 reduce 方案：先 reduce 到 rank 0，再 broadcast（不如 all_reduce 高效）

## 补充 B：Sequence Parallelism（SP）

### B.1 问题：TP 中的冗余计算

在标准 TP 中，attention 和 MLP 的 Linear 被切分了，但 LayerNorm、Dropout 等操作**在每个 rank 上做的是相同的完整计算**：

```
[TP rank 0] LayerNorm(full_hidden) → Column Linear(shard) → ...
[TP rank 1] LayerNorm(full_hidden) → Column Linear(shard) → ...
```

问题：
1. LayerNorm 的 activation（完整 hidden_size）在每个 rank 上都存了一份 → 浪费显存。
2. Dropout 在每个 rank 上都存了完整的 mask → 浪费。

### B.2 SP 的解决方案

Sequence Parallelism 把**非 TP 区域**（LayerNorm、Dropout）的 activation 沿 sequence 维度切分：

```
标准 TP:
  每个 rank 存完整 activation [batch, seq, hidden]

SP + TP:
  非 TP 区域：每个 rank 存 [batch, seq/tp, hidden]（按 sequence 切）
  TP 区域：每个 rank 存 [batch, seq, hidden/tp]（按 hidden 切，和 TP 一样）
```

### B.3 SP 的通信

在 TP region 和 non-TP region 的边界需要通信：

```
Non-TP → TP (Column Linear forward):
  之前：all_gather（把 seq 分片拼成完整 seq，给 Column Linear 用）
  SP 优化：用 all_gather 替代原来的 identity（通信量不变，但 activation 内存省了）

TP → Non-TP (Row Linear forward → LayerNorm):
  之前：all_reduce（Row 的 partial output 求和）
  SP 优化：用 reduce_scatter 替代 all_reduce（输出变成 seq 维度的分片）
```

### B.4 SP 的收益

| 方面 | 无 SP | 有 SP |
|---|---|---|
| LayerNorm activation 内存 | 每个 rank 存完整 [B, S, H] | 每个 rank 存 [B, S/tp, H] |
| Dropout mask 内存 | 完整 | 1/tp |
| 通信量 | 和纯 TP 相同 | 相同（reduce_scatter 替代 all_reduce，通信量一样） |
| 额外代价 | 无 | 实现复杂度 |

**结论**：SP 在不增加通信量的情况下，把非 TP 区域的 activation 内存从 O(S) 降到 O(S/tp)。对于长序列训练，这可以显著降低内存压力。

### B.5 SP vs CP（Context Parallelism）

| 方面 | SP | CP (Ring Attention) |
|---|---|---|
| 切什么 | 非 TP 区域的 activation，沿 seq 维度 | Attention 的 KV，沿 seq 维度 |
| 解决什么 | TP 中冗余 activation 的显存 | 长序列 attention score 矩阵的显存 |
| 通信方式 | all_gather / reduce_scatter（和 TP 配合） | Ring send/recv KV chunks |
| 适用场景 | TP 内部优化 | 超长序列（16K+） |
| 能否叠加 | 是 | 是，SP + CP 可以同时用 |

SP 和 CP 解决不同层面的问题，实际中可以同时使用。
