# L02 讲义：PyTorch 显存账本与训练 step 证据链

这一讲进入训练系统的第一层：单进程 PyTorch 训练。

训练脚本 OOM 时，很多同学第一反应是缩 batch 或换更小模型；step 变慢时，第一反应是说 GPU 慢。这样的判断缺少证据。训练系统排查要先把显存和时间拆成可解释部分：参数、梯度、optimizer state、activation、dataloader、forward、backward、optimizer。每一项都有明确的输入、状态和输出。

本讲的 patch 只实现三个显存计数函数。代码很小，但它建立了后续 FSDP、ZeRO、Megatron distributed optimizer、activation recompute 和 checkpoint 课程都会复用的基本账本。

## 1. 这节课学完要能回答什么

学完这一讲，你应该能回答下面这些问题。

1. 一个 PyTorch 模型的参数字节数如何精确计算？
2. `.grad` 什么时候存在，什么时候是 `None`？
3. plain SGD、SGD momentum、Adam/AdamW 分别会产生什么 optimizer tensor state？
4. 为什么 7B 参数模型用 Adam 训练时，静态显存远大于参数本身？
5. activation 显存为什么和 batch size、sequence length、层数、checkpointing 有关？
6. 一个训练 step 应该怎样拆成 dataloader、forward、backward 和 optimizer？
7. profiler trace 能证明什么，不能证明什么？
8. tiny trainer 的 step 证据如何映射到 Megatron-shaped `train_step`？

本讲属于 training systems。我们暂时不讲多卡通信，不讲 tensor parallel，也不讲 ZeRO/FSDP 的分片策略；这些内容都要建立在单进程显存账本之上。

## 2. 从一个 OOM 问题讲起

假设你要训练一个 7B 模型，参数使用 bf16。只看参数：

```text
7B parameters * 2 bytes = 14 GB
```

如果据此判断 24 GB GPU 可以训练，就会很快 OOM。训练时还有：

- gradients：通常和参数同形。
- optimizer states：Adam/AdamW 通常保存一阶矩和二阶矩。
- master weights：某些混合精度训练会保留 fp32 主权重。
- activations：forward 中为 backward 保留的中间张量。
- temporary buffers：kernel workspace、通信 buffer、allocator fragmentation 等。

所以“模型参数能放下”和“训练能跑”是两个问题。L02 先解决静态账本的可计算部分，再通过 tiny transformer smoke 观察 step 中的时间和 peak memory。

## 3. 参数显存：numel 乘 element size

PyTorch 参数是 `nn.Parameter`，本质上是 tensor。参数字节数的最小准确公式是：

```python
sum(p.numel() * p.element_size() for p in model.parameters())
```

**定义：**
`numel()` 是 tensor 元素个数，`element_size()` 是单个元素占用字节数。fp32 通常是 4 字节，fp16/bf16 通常是 2 字节。

**输入：**
`model.parameters()` 迭代得到的每个 parameter。

**中间状态：**
每个 parameter 单独计算 `numel * element_size`。混合 dtype 模型必须按每个 tensor 自己的 dtype 计算，不能假设全模型同一 dtype。

**输出：**
所有 parameter 字节数的整数和。

**代价和边界：**
这个数字只覆盖参数本身。它不包含 grad、optimizer state、activation、buffer、module buffer 或 CUDA allocator 已缓存但未释放的内存。

本讲测试 `test_works_with_mixed_dtype` 专门覆盖 fp16 + fp32 混合模型，防止学生把 dtype 写死。

## 4. 梯度显存：backward 后才有 `.grad`

在 PyTorch 中，参数的梯度通常保存在 `p.grad`。在第一次 backward 之前，`p.grad` 是 `None`。执行：

```python
loss.backward()
```

之后，参与计算图并需要梯度的参数会得到与参数同形的 grad tensor。对 fp32 模型来说，grad bytes 通常等于 param bytes。

**输入：**
`model.parameters()` 中每个 parameter 的 `.grad` 字段。

**中间状态：**
如果 `p.grad is None`，跳过它。如果存在，计算 `p.grad.numel() * p.grad.element_size()`。

**输出：**
所有 grad tensor 的字节数总和。

**代价和边界：**
`optimizer.zero_grad(set_to_none=True)` 会把 grad 置为 `None`，这时 `count_grad_bytes()` 应该返回 0。`zero_grad(set_to_none=False)` 可能保留全 0 tensor，显存语义不同。真实训练里还要考虑梯度累积、gradient bucket、DDP all-reduce buffer 和混合精度 scaler。

这就是为什么测试既检查 backward 后的 grad bytes，也检查 grad 为 `None` 时返回 0。

## 5. Optimizer state：SGD 和 Adam 差异很大

optimizer state 存在 `optimizer.state` 里。它是一个 dict，key 通常是 parameter，value 是该参数对应的状态 dict。

本讲只统计状态 dict 里的 tensor：

```python
for state in optimizer.state.values():
    for value in state.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
```

不同 optimizer 的状态差异很大。

| Optimizer | 常见 tensor state | 近似静态显存 |
|---|---|---|
| SGD，无 momentum | 无 tensor state | 0 |
| SGD + momentum | `momentum_buffer` | 约 1 倍参数 |
| Adam / AdamW | `exp_avg`、`exp_avg_sq` | 约 2 倍参数 |

“约”这个词要保留，因为某些 optimizer 会有 step tensor、fp32 master weight、factored state 或框架特定状态。测试允许 5% slack，就是为了不把少量 per-param scalar tensor 当成失败。

## 6. 用静态账本估算训练显存

对一个 fp32 模型，单进程 Adam 训练的静态账本可以粗略写成：

```text
params: 1x
grads: 1x
optimizer state: 2x
total static: 4x parameter bytes
```

对 bf16 参数训练，情况取决于 optimizer 和 master weight 策略。一个常见近似：

```text
params bf16: 2 bytes * N
grads bf16 or fp32: depends on training stack
Adam states fp32: 8 bytes * N  # exp_avg + exp_avg_sq
master params fp32: optional 4 bytes * N
```

这说明性能和显存表述必须带条件。不能只说“bf16 省一半显存”。如果 optimizer state 仍是 fp32，静态账本的主要部分可能仍然来自 optimizer。

## 7. Activation 显存：参数账本之外的动态部分

activation 是 forward 过程中为 backward 保留的中间张量。Transformer 训练里，activation 显存通常受这些条件影响：

- batch size
- sequence length
- hidden size
- layer count
- attention 实现
- 是否开启 activation checkpointing
- dtype

activation checkpointing 的直觉是少存一部分中间结果，在 backward 时重算。它的收益是降低 activation peak，代价是 backward 计算时间增加。

本讲 patch 不统计 activation。原因很简单：activation 不是静态地挂在 model 或 optimizer 上，它取决于具体输入、forward graph、autocast、checkpointing 和运行时 kernel。我们通过 `train_tiny_transformer.py` 的 peak memory 和 profiler trace 观察它。

## 8. 训练 step 的时间分解

`train_tiny_transformer.py` 把一个 step 拆成四段：

```text
dataloader
  -> device copy
  -> zero_grad
  -> forward
  -> loss
  -> backward
  -> grad clip
  -> optimizer.step
```

它记录这些字段：

| 字段 | 含义 |
|---|---|
| `step_time_ms` | 一个训练 step 的端到端时间 |
| `dataloader_time_ms` | 取 batch 的时间 |
| `forward_time_ms` | forward 和 loss 计算时间 |
| `backward_time_ms` | backward 和 grad clip 时间 |
| `optimizer_time_ms` | optimizer step 时间 |
| `tokens_per_sec` | 本 step token 数除以 step 时间 |
| `peak_memory_gb` | CUDA peak memory，CPU 时为 0 |
| `grad_norm` | grad clip 前的 norm |

性能结论必须带比较对象和条件。例如：

```text
在同一台 4090、batch_size=4、seq_len=512、bf16 autocast 打开的条件下，activation checkpointing 让 peak memory 从 A 降到 B，但 backward_time_ms 从 C 增到 D。
```

没有条件的“更快”“更省显存”没有排查价值。

## 9. Profiler trace 该怎么看

`maybe_profile()` 在 warmup forward 上导出 `artifacts/profiler/trace.json`。这个 trace 可以用 Chrome trace viewer 或 PyTorch profiler 工具打开。

它能帮助你回答：

- CPU 时间主要花在哪些 op 或 Python 调用。
- CUDA 时间是否集中在 matmul、attention、layernorm 或 copy。
- forward 是否被 dataloader 或 CPU 调度拖住。
- 是否存在异常长的单个 op。

它不能直接证明：

- 完整训练 step 的长期吞吐。
- 多卡通信瓶颈。
- 不同 batch/seq_len 下的通用结论。
- 没有 warmup 的第一次运行性能。

因此 profiler 要和 `metrics.jsonl`、配置文件、硬件信息一起读。

## 10. 从 Tiny Trainer 到 Megatron train_step

MiniInfra 有两条训练主线。

第一条是 `mini_infra/training/trainer.py`。它提供一个最小 trainer：

```text
run
  -> simulated or torch backend
  -> model forward
  -> loss
  -> backward
  -> optimizer.step
  -> metrics.jsonl
  -> checkpoint
```

第二条是 `mini_infra/megatron/training/training.py`。它保留 Megatron-shaped 生命周期：

```text
optimizer.zero_grad
  -> forward_backward_func
  -> parse losses
  -> optimizer.step
  -> lr_scheduler.step
  -> metrics
  -> training_log
  -> save_checkpoint
```

L02 的显存账本在这两条主线里都适用。真实 Megatron 会增加 pipeline schedule、tensor/data parallel、distributed optimizer、checkpoint 分片和通信重叠，但 `train_step` 的基本证据仍然是 loss、grad、optimizer、lr、metrics 和 checkpoint。

## 11. Lab 只验收哪一个最小合同

本讲 patch 只验收三个函数：

```python
count_param_bytes(model)
count_grad_bytes(model)
count_optimizer_state_bytes(optimizer)
```

它不验收 activation，也不验收 profiler，也不验收真实 GPU peak memory。这样设计是为了让学生先掌握最小静态账本。

完成 patch 后，要跑 smoke 或 notebook，把静态账本放回完整训练 step 中看：

- 静态账本解释了多少显存？
- peak memory 比静态账本多多少？
- 多出来的部分是否可能来自 activation？
- step 时间主要花在哪一段？

## 12. 生产排查入口

### 12.1 OOM

先看：

- 参数 bytes。
- grad bytes。
- optimizer state bytes。
- batch size、seq_len、dtype、activation checkpointing。
- `peak_memory_gb`。
- 是否存在梯度累积、DDP bucket、通信 buffer。

如果静态账本已经接近显存上限，先考虑 sharding、offload、optimizer state 压缩或换 optimizer。若静态账本很小但 peak 很高，优先看 activation 和临时 buffer。

### 12.2 step 变慢

先看：

- `dataloader_time_ms`
- `forward_time_ms`
- `backward_time_ms`
- `optimizer_time_ms`
- profiler trace
- 是否开启 checkpointing 或 autocast
- GPU 利用率和 CPU 数据准备

如果 dataloader 时间高，先查数据读取、sleep、worker 和 pin memory。若 backward 时间高，查 activation checkpointing、attention 实现、grad clip 和 autograd graph。

### 12.3 optimizer state 异常

先看：

- optimizer 类型。
- 是否已经执行过 `optimizer.step()`。
- state dict 里 tensor entry 的 dtype 和 shape。
- 是否有 master weight 或框架额外 state。

Adam 的 state 通常在第一次 step 后才完整出现。刚创建 optimizer 时 state 可能为空。

## 13. 小结

L02 的主线是把训练显存和 step 时间拆成可证据化的字段。参数、梯度和 optimizer state 可以用静态函数精确计算；activation 和临时 buffer 要通过运行时指标和 profiler 观察。这个边界清楚后，后续讲 FSDP、ZeRO、Megatron、activation recompute 和 checkpoint 时，学生才能判断每种技术到底减少了哪一类显存，增加了哪一类代价。
