# L04 讲义：ManualDDP 与梯度同步

## 0. 本讲目标

学完这一讲，你应该能完成五件事：

- 解释数据并行训练里 local grad、all-reduce sum、average grad 和 optimizer step 的关系。
- 说清楚 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、process group 和 backend 各自管什么。
- 手写一个最小 `ManualDDP.synchronize_grads()`，覆盖 no-op、跳过 frozen 参数、跳过 `grad=None`、all-reduce 和平均。
- 用 patch tests 判断梯度是否和 PyTorch DDP 对齐。
- 用 smoke artifact 区分真实 process group 证据和 fallback 结果。

这节课在 L02 单进程训练 step 之后。L02 已经把 forward、loss、backward、optimizer 和显存账本串起来；本讲只推进一个问题：当训练从一个进程变成多个进程后，每个 rank 怎样在 step 前拿到同一份平均梯度。

## 1. 真实问题：多 rank 的梯度会自然分叉

数据并行的基本假设是：每个 rank 持有同一份模型参数，但处理不同数据 shard。rank 0 看到一批样本，rank 1 看到另一批样本。两个 rank 的 forward loss 通常不同，backward 得到的 `param.grad` 也通常不同。

如果两个 rank 直接各自调用 `optimizer.step()`，模型副本会在下一步开始时分叉。这个错误有时不会立刻报错，因为每个进程里的 PyTorch 图都是合法的；它会表现成训练曲线和单卡大 batch 对不上，或者后续 checkpoint 的参数在不同 rank 上不一致。

DDP 要解决的是 step 前的梯度一致性。只要满足三个条件：

1. 初始参数一致。
2. 每个 rank 使用相同优化器配置。
3. 每个参数在 `optimizer.step()` 前看到同一份平均梯度。

那么每个 rank 独立执行 optimizer 后，参数仍然一致。

对于两个 rank，某个参数的本地梯度分别是 `g0` 和 `g1`。DDP 的目标输出是：

```text
average_grad = (g0 + g1) / 2
```

对于 `world_size = N`，目标输出是：

```text
average_grad = (g0 + g1 + ... + gN-1) / N
```

这个公式是本讲的主线。后面的 process group、all-reduce、barrier、patch test 和 smoke 都在验证这条链路。

## 2. 先把通信边界说清楚

### Rank

rank 是当前进程在通信组里的编号。全局 rank 常用来决定一个进程在 collective 中的位置；local rank 常用来选择本机设备，例如 `cuda:0` 或 `cuda:1`。

输入：

- 环境变量里的 `RANK`。
- 由 launcher 传入的进程编号。

输出：

- 当前进程参加 collective 时的身份。
- 日志和 artifact 中用于定位具体进程的字段。

边界：

- global rank 和 local rank 不能混用。local rank 只说明本机设备序号，不说明它在全局通信组里的编号。

### World Size

`world_size` 是同一个 process group 里参与通信的进程数。梯度平均要除以它，smoke 里也用它判断应该启动几个进程。

输入：

- 环境变量里的 `WORLD_SIZE`。
- `dist.get_world_size(process_group)` 的返回值。

输出：

- 梯度平均的除数。
- collective 参与者数量。

边界：

- `world_size = 1` 时没有跨进程同步需求，`synchronize_grads()` 应直接返回。
- 分布式未初始化时也应直接返回，否则单进程 patch test 会被通信 API 卡住。

### Process Group

process group 是 collective 的参与者集合。`dist.all_reduce(..., group=process_group)` 只要求这个 group 里的 rank 进入同一次通信。

输入：

- backend，例如 `gloo` 或 `nccl`。
- master address、port、rank、world size。

中间状态：

- 每个进程持有同一个 group 的通信上下文。

输出：

- all-reduce、barrier 等 collective 可以在这个 group 内执行。

边界：

- group 初始化成功只能证明通信上下文存在，不能证明梯度平均正确。
- 所有 rank 必须以相同顺序进入相同 collective。一个 rank 少调一次 all-reduce，另一个 rank 仍在等待时，常见结果是 timeout 或 hang。

### Backend

backend 是 process group 底层通信实现。CPU 测试通常用 `gloo`；GPU 多卡训练通常用 `nccl`。本讲 patch tests 使用 CPU/gloo，因为它足以验证梯度平均语义；NCCL 性能、拓扑和带宽不在本讲 patch 范围内。

## 3. All-Reduce 到底做了什么

all-reduce 是集合通信。每个 rank 输入一个 tensor，通信库按指定操作规约所有输入，再把规约结果发回每个 rank。

用 `SUM` 举例：

```text
rank0 input: 1
rank1 input: 2

all_reduce(SUM)

rank0 output: 3
rank1 output: 3
```

它的输入是每个 rank 的 tensor；中间状态是通信库对所有 rank 的 tensor 做规约；输出是每个 rank 都得到同一份规约结果。

关键点是：`dist.all_reduce(..., op=SUM)` 只求和，不会自动求平均。DDP 梯度平均还需要除以 `world_size`。

```python
dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.process_group)
p.grad /= self.world_size
```

如果忘记除以 `world_size`，梯度会被放大。假设 `world_size = 8`，等效更新幅度接近使用 `8 * lr`。训练可能仍能跑几步，但它已经偏离你以为的学习率和 batch 语义。

先除再 all-reduce 也能得到同样结果：

```text
sum(g_i / world_size) == sum(g_i) / world_size
```

本讲参考实现采用“先 all-reduce sum，再除以 world size”，因为它和白板公式最直接对应。

## 4. ManualDDP 的最小合同

本讲不直接使用 `torch.nn.parallel.DistributedDataParallel`。我们先写一个教学版 `ManualDDP`，把 DDP 最核心的数学合同留下来，把 bucket、hook、通信计算重叠和容错暂时拿掉。

### 构造函数

`ManualDDP.__init__` 需要保存三项状态：

```python
self.module = model
self.process_group = process_group
self.world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
```

这里不要复制模型参数。`self.module.parameters()` 应该和传入的 `model.parameters()` 指向同一批参数对象。测试会依赖这个行为。

输入：

- 原始 `nn.Module`。
- 可选 `process_group`。

输出：

- 一个持有原模型引用和同步边界的 wrapper。

边界：

- 构造函数不启动通信。
- forward 不通信。
- optimizer 不由 `ManualDDP` 管理。

### synchronize_grads

`synchronize_grads()` 的输入是已经完成 backward 的模型。此时每个 rank 的 `p.grad` 里放着 local grad。函数要原地把 local grad 改成 average grad。

执行顺序：

1. 如果 `world_size == 1` 或 `dist` 未初始化，直接返回。
2. 遍历 `self.module.parameters()`。
3. 如果 `p.requires_grad` 为 false，跳过。
4. 如果 `p.grad is None`，跳过。
5. 对 `p.grad` 执行 `dist.all_reduce(..., SUM, group=process_group)`。
6. 原地执行 `p.grad /= self.world_size`。

中间状态：

- all-reduce 后，所有 rank 的 `p.grad` 都是梯度和。
- 除法后，所有 rank 的 `p.grad` 都是平均梯度。

输出：

- 每个 rank 的有效参数都持有相同平均梯度。

边界：

- frozen 参数没有梯度，不应参与通信。
- 某些分支未使用时，参数可能出现 `grad=None`，同步函数不能崩溃。
- 本讲没有处理真实 DDP 里的 unused parameter 检测、bucket rebuild 和 autograd hook。

## 5. 为什么跳过参数也是通信合同的一部分

很多同学第一次写 DDP 会把问题简化成“遍历所有参数并 all-reduce”。这会漏掉两个边界。

第一类是 frozen 参数。`requires_grad=False` 的参数不会由 autograd 产生 grad。对这类参数通信没有意义，强行访问 `p.grad` 还可能遇到 `None`。

第二类是 partial grad。模型存在条件分支、dead path 或局部冻结时，某些参数在本次 backward 中没有参与 loss 计算，`p.grad` 会是 `None`。本讲 patch test 会手动把一个参数的 grad 设为 `None`，检查同步函数是否能跳过它。

这里还藏着一个真实系统风险：所有 rank 的 collective 顺序必须一致。如果 rank 0 对参数 A 发起 all-reduce，rank 1 却跳过参数 A 并对参数 B 发起 all-reduce，两个 rank 就进入了不同 collective。真实 DDP 用 reducer、bucket、unused parameter 检测来管理这些复杂情况。本讲的模型和测试保持简单，目的是先让你看清 skip 的基本边界。

## 6. Patch Tests 验证什么

本讲 patch 命令是：

```bash
make patch-test M=l03_nccl_ddp_smoke
```

它会运行 5 个 CPU/gloo 测试。它们验证的是 `ManualDDP` 的数值语义和边界行为，不验证 NCCL 带宽，也不验证多机稳定性。

| 测试 | 验证点 |
|---|---|
| `test_grads_match_pytorch_ddp` | 和 PyTorch DDP 在相同输入下的梯度一致 |
| `test_grads_are_averaged_not_summed` | 结果是平均梯度，不能差一个 world size 因子 |
| `test_world_size_1_is_noop` | 单进程时同步函数不改 grad |
| `test_skips_no_grad_params` | frozen 参数不参与通信 |
| `test_handles_partial_grads` | `grad=None` 参数不会让同步函数报错 |

这些测试的比较对象很明确：PyTorch DDP、单进程 baseline、world size 为 1 的原始 grad、frozen 参数和 partial grad。通过这些测试后，你能说明 `ManualDDP` 的最小合同成立。

## 7. Smoke 证据链怎么读

patch test 关注梯度同步逻辑。smoke 关注最小 process group 和 all-reduce 链路。命令是：

```bash
python labs/l03_nccl_ddp_smoke/scripts/run_smoke.py --mode smoke
```

它会尝试运行：

```text
run_smoke.py
  -> torch.distributed.run --standalone --nproc_per_node=2
  -> ddp_hello.py
  -> artifacts/ddp_hello.json
  -> metrics.jsonl
  -> report.md
```

`ddp_hello.py` 的最小实验是：每个 rank 构造一个值为 `rank + 1` 的 tensor。2-rank 情况下，rank 0 输入 1，rank 1 输入 2。经过 `all_reduce(SUM)` 后，两个 rank 都应得到 3。脚本随后执行 barrier，证明所有 rank 到达同一个同步点。

关键字段：

| 字段 | 解释 |
|---|---|
| `world_size` | 本次 process group 的进程数 |
| `backend` | 使用 `gloo`、`nccl` 或 fallback |
| `all_reduce_sum` | rank+1 tensor 求和后的结果，2-rank 时应为 3 |
| `barrier_ok` | 是否到达 barrier |
| `fallback_used` | 是否走模拟结果 |

`fallback_used=True` 时，本次结果只能说明脚本产出了 validation artifact。它不能当作真实 process group、NCCL 或 DDP 通信通过的证据。报告里必须把这个边界写清楚。

## 8. 生产 DDP 多了哪些复杂度

`ManualDDP` 把通信放在 backward 完成之后。这样教学清楚，但性能路径落后于生产 DDP。

真实 PyTorch DDP 会在 backward 过程中通过 autograd hook 得知哪些梯度已经 ready，再把梯度放入 bucket。某个 bucket 满足同步条件后，DDP 可以提前发起 all-reduce。这样一部分通信时间能和剩余 backward 计算重叠。

粗略看 step 时间：

```text
no overlap:     T_step ≈ T_backward_compute + T_grad_comm + T_optimizer
good overlap:   T_step ≈ max(T_backward_compute, T_grad_comm) + T_optimizer
```

这个公式只用于解释时间来源。真实收益取决于模型层数、梯度总字节数、bucket size、网络拓扑、GPU/NIC 带宽、batch size、梯度累积策略和 profiler 观察到的 overlap 程度。小模型或 batch 很小时，通信启动开销可能盖过并行收益。

真实训练还会遇到：

- gradient accumulation：前几个 micro-step 常用 `no_sync()` 跳过同步，最后一个 micro-step 再同步。
- unused parameters：不同分支可能导致部分参数本轮没有 grad。
- buffer broadcast：BatchNorm 等 buffer 可能需要额外同步。
- failure handling：某个 rank 退出会影响整个 group。
- bucket tuning：bucket 太小会增加通信启动次数，bucket 太大会推迟通信开始。

这些都是 L02 之后继续展开的内容。本讲只要求你把平均梯度这条底层合同讲准、写对、测明白。

## 9. 本讲小结

这节课从一个具体问题开始：多个 rank 各自 backward 后，local grad 不同，直接 step 会让模型副本分叉。DDP 的最小数学合同是把每个参数的 local grad 做 `all_reduce(SUM)`，再除以 `world_size`，使每个 rank 在 step 前持有同一份平均梯度。

`ManualDDP` 的实现刻意朴素：forward 不通信，构造函数不复制参数，`synchronize_grads()` 只处理 no-op、skip、all-reduce 和平均。patch tests 验证它和 PyTorch DDP 的梯度语义对齐；smoke artifact 验证最小 process group、all-reduce 和 barrier 链路。看到 fallback 时，要把它写成 validation-only。进入后续课程前，你需要能把这条证据链从源码、测试和产物三处同时讲清楚。
