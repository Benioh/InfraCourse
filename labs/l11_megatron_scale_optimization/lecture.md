# L12：Megatron Scale Optimization 与 Bucketed Manual DDP

L12 讨论训练扩展里最常见的 data parallel 梯度通信问题。L11 已经把 `train_step` 的调用顺序接回训练闭环；现在关注 backward 结束后到 optimizer step 前的同步环节：每个 rank 拿到本地梯度后，怎样用更少的 collective 调用得到跨 rank 平均梯度。

本关 patch 实现同步式 `BucketedManualDDP`。它不调用 `torch.nn.parallel.DistributedDataParallel`，也不实现 autograd hook、异步 overlap、`gradient_as_bucket_view`、distributed optimizer 或 Megatron 的 reduce-scatter。先把 bucket 构造和数值等价写对，再把这条主线映射到生产 DDP。

## 1. 本讲目标：把逐参数 grad sync 改成按桶通信

学完本讲后，你应该能完成五件事：解释很多小参数逐个 all-reduce 为什么会浪费启动开销；按 `requires_grad=True` 参数顺序和 `bucket_size_mb` 构造 bucket；用 `_flatten_dense_tensors` 和 `_unflatten_dense_tensors` 保持 shape 与数值等价；处理 world_size=1、超大单参数和 frozen param；把本地实现对照到 Megatron `ParamAndGradBuffer`、bucket group、overlap grad reduce 和 distributed optimizer。

本关的输入是一个 PyTorch module、bucket size 和可选 process group。初始化阶段输出 `self.buckets`；反向传播后，`synchronize_grads()` 读取每个 bucket 中已有的 `p.grad`，合并、all-reduce、除以 world size，再 copy 回原梯度。测试用 2-rank gloo CPU 验证结果和 PyTorch DDP 一致。

直观理解：逐参数通信像每张纸单独寄出；bucket 通信像把一摞纸装进同一个信封，收件后再按原顺序分发。

为什么要学：DDP 的通信瓶颈不只由总字节数决定。collective 次数、启动开销、buffer 连续性和 backward overlap 都会影响 step time 和 MFU。

常见误解：

- 认为 bucket 会减少总通信字节；它主要减少 collective 启动次数。
- 把 bucketed DDP 当成 ZeRO；本关仍让每个 rank 持有完整参数和完整梯度。
- 以为 CPU/gloo 测试能代表 NCCL throughput；本关只验证语义。

自检问题：

- BucketedManualDDP 相比逐参数 ManualDDP 改了哪一步？
- bucket 合并主要减少的是通信字节还是 collective 启动次数？
- 本关为什么禁止直接使用 PyTorch DDP？

## 2. 问题背景：小 collective 太多会拉低通信效率

all-reduce 的耗时可以粗略拆成固定启动开销和按字节传输的带宽开销。假设一次 collective 启动约 10 微秒，模型有 1000 个参数张量，逐参数 all-reduce 会先花约 10 毫秒在启动上。参数越碎，这部分开销越明显；总字节数相同，通信次数不同，step time 会差很多。

bucket 的做法是把多个 grad 拼成一个连续 tensor，让一次 collective 覆盖一组参数。这样总字节数基本不变，NCCL 或 gloo 处理的是更大的连续 buffer，启动开销被摊到更多字节上。真实 PyTorch DDP 和 Megatron 还会让某个 bucket 的梯度一凑齐就发起异步通信，把通信和后续 backward 计算重叠起来。

bucket size 是 trade-off。太小，collective 次数仍多；太大，最早的 bucket 要等更多梯度准备好，overlap 机会减少，单次通信也更长。本关默认 25MB 是常见经验值，但 patch 测试会用很小的 bucket size 强制产生多个桶，方便检查逻辑。

自检问题：

- all-reduce 时间里的启动开销和带宽开销分别是什么？
- 为什么很多小参数会放大启动开销？
- bucket size 过小和过大分别有什么问题？

## 3. 核心概念一：按字节数构造 bucket

bucket 是一组需要同步梯度的参数列表。本关按 `self.module.parameters()` 顺序遍历，只收集 `requires_grad=True` 的参数，用 `p.numel() * p.element_size()` 估算字节数。当前桶加上下一个参数超过 `bucket_size_bytes` 时，先结束当前桶，再新开一桶。

每个参数可以理解成一件不同大小的包裹。当前箱子还能装就继续放，放不下就封箱；如果单件包裹已经比箱子大，它自己占一个箱子，不能把一个参数切开。

bucket 划分要稳定。测试会检查超大参数、很多小参数和 frozen parameter。实现如果把 frozen param 放进 bucket，后面会遇到 `p.grad is None`；如果强行按 bucket size 切单个参数，就破坏了参数到 grad 的 shape 对应关系。

常见误解：

- 按参数大小排序，导致调试时 bucket 顺序和模型顺序对不上。
- 把 `requires_grad=False` 参数也放进 bucket。
- 试图切分一个超大参数来满足 bucket size，破坏本关接口契约。

自检问题：

- `bucket_size_mb` 如何换算成 `bucket_size_bytes`？
- 单个参数大于 bucket size 时应该怎么处理？
- 为什么 frozen param 不应该进入 bucket？

## 4. 核心概念二：flatten、all-reduce、unflatten 保持数值等价

`_flatten_dense_tensors(grads)` 会把一组 dense tensor 展平成一个连续 tensor；`_unflatten_dense_tensors(flat, grads)` 会按原 tensor 的 shape 和 numel 切回一组 tensor。本关对每个 bucket 执行 flatten、all-reduce、除以 world size、unflatten、copy 回原 grad。

数值等价来自两个条件：第一，所有 rank 对同一参数位置做 sum 后除以 world size；第二，unflatten 后 copy 回原 grad 时顺序和 shape 不变。合并通信不改变梯度数学，只改变通信批量。

all-reduce 后必须除以 world size，才能和 PyTorch DDP 默认的平均梯度对齐。本关测试比较的是平均后的 grad，不是 raw sum。

常见误解：

- flatten 后直接替换 `p.grad` 对象，导致 optimizer 或外部引用失效。
- 忘记除以 world size，得到 sum gradient。
- 对 `grad is None` 的参数也参与 flatten，导致运行时错误。

自检问题：

- flatten 和 unflatten 分别解决什么问题？
- 为什么 all-reduce 后要除以 world size？
- copy 回原 grad 时为什么要保持 bucket 内顺序？

## 5. 机制讲解：synchronize_grads 的输入、状态和输出

机制输入是已经完成 backward 的 model 参数梯度、`self.buckets`、`self.world_size` 和 `process_group`。中间状态是当前 bucket 的 `grads` 列表、flatten 后的 `flat` tensor、all-reduce 后的平均梯度、以及 unflatten 得到的 per-parameter grad。输出没有新对象返回，副作用是每个参数的 `p.grad` 被替换为跨 rank 平均后的梯度。

执行过程先处理边界：如果 `world_size == 1` 或 `torch.distributed` 没初始化，直接返回。然后逐个 bucket 收集非 None 的 grad；空 bucket 跳过；非空 bucket 先 flatten，再调用 `dist.all_reduce(flat, op=SUM, group=process_group)`，随后 `flat /= world_size`，最后用 unflatten 结果 `copy_` 回原来的 grad tensor。

这个机制解决的是“多 rank 梯度一致”问题。每个 rank 的输入 batch 可以不同，本地 backward 得到的梯度也不同；all-reduce 平均后，所有 rank 在 optimizer step 前看到同一组 averaged grads，因此后续参数更新保持一致。

自检问题：

- `synchronize_grads` 的输入、中间状态和输出分别是什么？
- 为什么 world_size=1 时应该直接返回？
- 本地梯度平均后，为什么所有 rank 的参数更新会保持一致？

## 6. 工程验证：2-rank gloo 测数值和边界

本关验证命令是：

```bash
make patch-test M=l11_megatron_scale_optimization
```

测试 harness 会用 multiprocessing 启动 1 或 2 个 gloo rank，设置 `MASTER_ADDR`、`MASTER_PORT`、`RANK` 和 `WORLD_SIZE`，然后在每个 worker 里导入 starter 或 reference。核心测试包括：与 PyTorch DDP 的 grad 对齐；单个超大参数超过 bucket size 时不崩；很多小参数能合到同一桶；world_size=1 不改 grad；冻结层参数不参与同步。

这些测试关注语义，不测真实性能。`test_works_with_tiny_params` 会把 10 个小 Linear 层放进大 bucket，验证合并后梯度仍和单 rank 参考一致；`test_grads_match_pytorch_ddp` 用不同 rank 输入对比 PyTorch DDP，验证平均语义。

自检问题：

- patch-test 为什么需要 2 个 rank？
- 哪个测试验证单个参数大于 bucket size 的情况？
- 为什么 frozen param 的 grad 应该保持 None？

## 7. 生产对照：Megatron 用连续 buffer、bucket group 和 overlap

Megatron 的真实 DDP 比本关复杂很多，但主线能对上。它先收集 trainable params，再按 dtype、grad dtype、expert parallel 等条件分组，给每组分配 `_ParamAndGradBuffer`。buffer 里有参数和梯度的连续存储，bucket 记录自己负责的参数列表、在大 buffer 中的 offset，以及 param 到局部区间的映射。

通信层也更复杂。Megatron 会把 bucket 放进 bucket group，用 backward hook 标记参数梯度 ready；当一个 bucket group 的梯度齐全时，可以异步发起 all-reduce 或 reduce-scatter。没有 distributed optimizer 时，它做 data parallel all-reduce；启用 distributed optimizer 时，它对 grad buffer 做 reduce-scatter，让每个 rank 只拿自己负责的 shard。

本关的手动实现对应生产路径里的最小子集：按 bucket 收集梯度、对连续 buffer 做 all-reduce、再写回梯度。它没有 main_grad、grad buffer view、异步 handle、stream 同步、reduce-scatter、param all-gather、fp32 accumulation 或 MoE expert data parallel。

自检问题：

- Megatron 为什么需要 ParamAndGradBuffer？
- bucket 和 bucket group 的职责有什么区别？
- distributed optimizer 路径为什么会使用 reduce-scatter？

## 8. 性能边界：bucket 只是 scale optimization 的一块

bucketed DDP 主要改善 grad sync 的 collective 组织。它不减少模型参数量，不减少总梯度字节，也不解决 activation 显存、optimizer state 显存或 pipeline bubble。真实 scale optimization 要同时看 TP、PP、DP、recompute、bucket size、checkpoint I/O、数据加载和网络拓扑。

课程里的 `run_scale_stub.py` 用一个估算表把这些指标放在一起：TP 会降低单卡模型显存但增加通信惩罚；PP 会降低部分激活压力但引入 bubble；activation recompute 会降低激活显存但增加计算；GPU 数增加会提高 tokens/sec，也可能暴露通信或 I/O 新瓶颈。报告里不能只写“更快”，要写比较对象、指标、配置和代价。

排障也要按证据走。MFU 低时先对照 `command.sh`、`config.resolved.yaml`、主日志和 `metrics.jsonl`；recompute 取舍错误时先改一个变量做小步实验；如果换 world size 或分片策略，还要考虑 checkpoint 里的 optimizer state layout。

自检问题：

- bucketed DDP 解决 scale optimization 中的哪一类瓶颈？
- TP、PP、recompute 各自改变哪些指标和代价？
- 低 MFU 排查为什么要先固定 command、config 和 metrics？

## Lab 验收边界

本讲 patch 命令：`make patch-test M=l11_megatron_scale_optimization`。

patch 验收的是：把逐参数 ManualDDP 升级成 `BucketedManualDDP`。实现应按字节数给 params 分桶，每个桶用 `_flatten_dense_tensors` 拼接后一次 all-reduce，再用 `_unflatten_dense_tensors` 拆回去，最终梯度与 PyTorch DDP 对齐。

---

## 补充：ZeRO-1 / ZeRO-2 / ZeRO-3 系统对比

### 概述

ZeRO（Zero Redundancy Optimizer）的核心思想：数据并行中每张卡都持有完整的 optimizer state、gradient 和 parameter 是冗余的。ZeRO 通过分片（partition）消除冗余，按分片粒度分为三个阶段。

### ZeRO-1：只分片 Optimizer State

**分片对象**：optimizer state（如 Adam 的 m 和 v，每个参数 2 份 fp32 副本）。

**通信模式**：梯度仍然用标准 all-reduce 同步（与普通 DDP 相同），每张卡拿到完整梯度后，只更新自己负责的那部分参数的 optimizer state 和参数值，最后用 all-gather 把更新后的参数广播给所有卡。

**显存节省公式**：

```
标准 DDP 单卡 optimizer 显存 = Ψ × K  （K=12 for Adam fp32: 4 bytes param + 4 bytes m + 4 bytes v）
ZeRO-1 单卡 optimizer 显存 = Ψ × K / N  （N = GPU 数）
```

对于 Adam mixed-precision（fp16 params + fp32 optimizer），optimizer state 占总显存的大头（约 12× model size in bytes），ZeRO-1 把这部分除以 N。

**适用场景**：通信带宽充裕、想最小改动获得显存收益。通信量与 DDP 相同。

### ZeRO-2：分片 Optimizer State + Gradient

**分片对象**：optimizer state 和 gradient 都按参数分片到不同卡。

**通信模式**：用 **reduce-scatter** 替代 all-reduce。backward 时每个参数的梯度通过 reduce-scatter 分发：每张卡只收到自己负责的那片参数的聚合梯度，其他片段的梯度即时释放。之后各卡用本地梯度更新本地 optimizer state 和参数，最后 all-gather 参数。

**显存节省公式**：

```
ZeRO-2 单卡显存 = Ψ × 2 (fp16 params + fp16 grads/N + optimizer/N)
                 ≈ 2Ψ + 2Ψ/N + 12Ψ/N
```

相比 ZeRO-1，额外节省了 gradient 显存（从 2Ψ 降到 2Ψ/N）。

**通信量**：reduce-scatter + all-gather = all-reduce，因此通信总量与 DDP 和 ZeRO-1 相同（2Ψ bytes per step），但 reduce-scatter 允许梯度在 backward 中流式释放，降低显存峰值。

### ZeRO-3：分片一切（Optimizer + Gradient + Parameter）

**分片对象**：optimizer state、gradient、parameter 全部分片。每张卡只持有 1/N 的参数。

**通信模式**：

- **Forward**：每层计算前 all-gather 该层参数 → 计算 → 释放非本地参数。
- **Backward**：每层 all-gather 参数 → 计算梯度 → reduce-scatter 梯度 → 释放非本地参数和非本地梯度。
- **Update**：每张卡只更新本地 1/N 参数。

**显存节省公式**：

```
ZeRO-3 单卡显存 = (2Ψ + 12Ψ + 2Ψ) / N = 16Ψ / N
加上临时 buffer（forward/backward 时需要完整单层参数）
```

理论上 N 足够大时，单卡模型相关显存趋近于 0，瓶颈变成激活显存。

**通信量**：每层 forward 一次 all-gather + backward 一次 all-gather + 一次 reduce-scatter = 3Ψ bytes per step（比 ZeRO-1/2 多 50%）。

### 通信代价对比表

| 阶段 | 分片内容 | 单卡显存（近似） | 通信量/step |
|---|---|---|---|
| DDP（无 ZeRO） | 无分片 | 16Ψ + activations | 2Ψ (all-reduce) |
| ZeRO-1 | optimizer state | 4Ψ + 12Ψ/N | 2Ψ (all-reduce) |
| ZeRO-2 | optimizer + grad | 2Ψ + 14Ψ/N | 2Ψ (reduce-scatter + all-gather) |
| ZeRO-3 | optimizer + grad + param | 16Ψ/N | 3Ψ (额外 all-gather for params) |

注：Ψ = 模型参数量 × 2 bytes（fp16），上表忽略激活显存。

### 选择建议

1. **ZeRO-1**：最简单，通信不变，适合 optimizer state 是显存瓶颈的场景（大模型 + 大 optimizer 如 Adam）。大多数情况下是默认起步选择。

2. **ZeRO-2**：梯度显存也成为瓶颈时升级。通信量不变但需要把 all-reduce 拆成 reduce-scatter + all-gather，对通信库有 bucket 支持要求。适合中等规模模型（7B-30B）。

3. **ZeRO-3**：模型大到单卡放不下参数时必须使用。通信增加 50%，且 forward/backward 都有额外 all-gather 延迟。适合超大模型（>30B）或 GPU 显存非常有限的场景。

4. **ZeRO-3 + offload**：进一步把 optimizer state offload 到 CPU，换取更低 GPU 显存但增加 CPU-GPU 通信。适合单机少卡训练超大模型（研究场景）。

实践中，FSDP（PyTorch）本质是 ZeRO-3 的实现；DeepSpeed 支持 ZeRO-1/2/3 全部阶段并可动态切换。选择时先 profile 显存分布：如果 optimizer 占大头用 ZeRO-1；如果参数本身放不下用 ZeRO-3；其余情况 ZeRO-2 是性价比最高的选择。
