# Debug Checklist：PyTorch 显存账本与训练 step

## 1. 先固定现场

- 保存命令、配置、git commit、Python/PyTorch/CUDA 环境。
- 记录 batch size、sequence length、hidden size、layer count、dtype、optimizer、activation checkpointing。
- 保存 `metrics.jsonl`、`train.log`、`artifacts/profiler/trace.json` 和 `report.md`。
- 明确这是 CPU smoke、本地 GPU smoke，还是更大模型训练。

## 2. OOM 排查

先做静态账本：

1. `param_bytes = sum(p.numel() * p.element_size())`
2. backward 后看 `grad_bytes`
3. optimizer step 后看 `optimizer_state_bytes`
4. 合计 params + grads + optimizer state

再看运行时条件：

| 条件 | 影响 |
|---|---|
| batch size | 直接影响 activation |
| sequence length | Transformer activation 和 attention 代价增长明显 |
| dtype | 参数、grad、activation 的 element size 可能变化 |
| activation checkpointing | 降低 activation peak，增加 backward 计算 |
| optimizer | Adam/AdamW state 通常大于 SGD |
| DDP/FSDP/ZeRO | 会改变 grad、optimizer state 和通信 buffer 边界 |

判断：

- 静态账本已经接近显存上限：先考虑换 optimizer、分片、offload 或缩模型。
- 静态账本远小于 peak memory：优先看 activation、temporary buffers、CUDA allocator 和 profiler trace。
- 开启 checkpointing 后 peak 降低但 backward 变慢：这是预期 tradeoff，需要量化收益和代价。

## 3. step 变慢排查

按 `metrics.jsonl` 逐项看：

1. `dataloader_time_ms`
2. `forward_time_ms`
3. `backward_time_ms`
4. `optimizer_time_ms`
5. `step_time_ms`
6. `tokens_per_sec`

判断：

- `dataloader_time_ms` 高：查数据读取、sleep、num_workers、pin memory、CPU 瓶颈。
- `forward_time_ms` 高：查 batch/seq_len、attention 实现、dtype、autocast、模型层数。
- `backward_time_ms` 高：查 activation checkpointing、grad clip、autograd graph、loss scale。
- `optimizer_time_ms` 高：查 optimizer state 大小、CPU/GPU state、参数数量。
- `tokens_per_sec` 波动大：查 dataloader、warmup、GPU clock、后台任务和输入 shape 是否变化。

## 4. optimizer state 异常

按顺序看：

1. optimizer 类型。
2. 是否执行过 `loss.backward()`。
3. 是否执行过 `optimizer.step()`。
4. `optimizer.state[p]` 中有哪些 key。
5. 每个 state tensor 的 dtype、shape 和 bytes。

判断：

- optimizer 刚创建时 state 为空是正常现象。
- plain SGD state 为 0 是正常现象。
- SGD momentum 约等于 1 倍参数。
- Adam/AdamW 通常约等于 2 倍参数，额外 master weight 要另算。

## 5. profiler trace 排查

看 trace 前先记录：

- 硬件和 CUDA 版本。
- batch size、sequence length、dtype。
- trace 覆盖的是 forward、完整 step，还是某个 warmup。
- 是否已经完成 warmup。

判断：

- 单个 op 很长：查输入 shape、dtype、kernel fallback。
- CPU 空白很大：查 Python 调度、dataloader、同步点。
- CUDA kernels 很碎：查小 batch、小 op 或过多 Python overhead。
- trace 只覆盖 warmup forward：不要写成完整训练吞吐结论。

## 6. 结束条件

复盘结束时要写清：

- 静态账本是多少 bytes/GB。
- peak memory 比静态账本多多少。
- step 时间主要花在哪一段。
- profiler 只证明了哪段路径。
- 下一步要改哪个变量，并预期影响哪个指标。
