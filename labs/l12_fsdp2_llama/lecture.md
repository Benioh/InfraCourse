# L13：FSDP2 与 Llama Block 分片训练

## 本讲目标

- 理解 Fully Sharded Data Parallel 2（FSDP2）解决的训练显存问题。
- 能解释 `fully_shard` 的 wrap 粒度、block-first/root-last 顺序和 reshard 代价。
- 能说明 `MixedPrecisionPolicy` 对参数计算、梯度通信和输出 dtype 的影响。
- 能读懂 TorchTitan 在 Llama 模型上应用 FSDP2 的主路径。
- 能用 patch-test、CPU dryrun 和 GPU train smoke 区分“wrap 逻辑正确”和“真实性能成立”。

## 1. 问题背景：Data Parallel 的显存重复

Data Parallel 把同一个模型复制到多个 rank 上，每个 rank 处理不同 batch shard，然后同步梯度。这个方案简单，但它让参数、梯度和 optimizer state 在每张卡上都保留一整份。模型从几亿参数走到 1B、7B 时，即使计算能跑，显存也会先被参数状态占满。

以 AdamW 训练为例，一个参数通常会对应训练权重、梯度、一阶动量、二阶动量，有时还有 fp32 master weight。Data Parallel 增加 rank 数可以提高数据吞吐，但不会减少单卡保存的模型状态。FSDP2 的目标就是把这些状态按 data parallel rank 切开，让每张卡只保存自己负责的 shard。

FSDP2 的代价也很直接。模型某个模块执行 forward 或 backward 时，通常需要临时拿到完整参数，因此 runtime 要做 all-gather；计算结束后如果启用 reshard，又会把完整参数释放回 shard。它用通信和调度复杂度换显存，不是免费加速。

**小检查：**

1. Data Parallel 增加 rank 数时，单卡参数显存会下降吗？
2. FSDP2 降低的是参数状态显存，还是激活显存？
3. 为什么 FSDP2 会引入额外 all-gather？

## 2. 核心概念一：FSDP2 的分片单元

Fully Sharded Data Parallel 2（FSDP2）是 PyTorch 的 composable 参数分片 API。它通过 `fully_shard(module, ...)` 把一个模块变成 FSDP 单元。这个单元内部的参数在 rank 间分片保存，需要完整参数时再临时 all-gather。

直观地说，FSDP2 不会把一整个训练脚本包成黑盒；它让你决定“哪些模块应该成为分片单元”。对 Llama-style 模型，常见做法是先 wrap 每个 transformer block，再 wrap root model。block 是主要计算单元，按 block 分片能让 runtime 在 block 粒度上 unshard 和 reshard；root wrap 负责 embedding、norm、head 等剩余参数。

wrap 粒度影响显存峰值和通信节奏。粒度太粗，例如只 wrap root，执行时可能需要更大范围的完整参数，显存峰值高；粒度太细，例如把很多小子层都单独 wrap，会增加调度和通信次数。课程 patch 采用 block 粒度，因为它和 TorchTitan 的 Llama 主路径一致，也足够解释生产训练里的关键边界。

**小检查：**

1. 为什么 Llama block 适合作为 FSDP2 wrap 单元？
2. 只 wrap root 会带来什么显存和通信问题？
3. wrap 粒度太细可能增加哪类开销？

## 3. 核心概念二：unshard、compute、reshard

FSDP2 单元执行时可以分成三个阶段。第一，runtime 在进入模块计算前 all-gather 参数 shard，得到本模块需要的完整参数视图。第二，模块执行 forward 或 backward。第三，如果配置允许，runtime 在 forward 后 reshard，把完整参数释放回 shard 状态，降低后续峰值显存。

`reshard_after_forward=True` 常用于显存紧张的训练。它的收益是 forward 后不继续持有完整参数；代价是 backward 需要这些参数时可能再次 all-gather。对于长序列、大 microbatch 或多层模型，这个选择可能决定是否 OOM。对于 pipeline parallel 场景，反复 reshard 可能和 microbatch 调度发生冲突，TorchTitan 因此把 reshard policy 抽成配置。

本讲 patch 只要求把 `reshard_after_forward` 原样传给每一次 `fully_shard`。它不模拟真实 all-gather，也不测通信 overlap。测试能证明你把配置传对了，不能证明某个 profile 在 H200 上最快。

**小检查：**

1. `reshard_after_forward=True` 为什么能降低显存？
2. 它为什么可能增加 backward 前的通信？
3. CPU patch-test 为什么不能证明真实 reshard 性能？

## 4. 核心概念三：MixedPrecisionPolicy

混合精度策略（Mixed Precision Policy）定义 FSDP2 模块内部的 dtype 规则。`param_dtype` 通常控制参数参与 forward 的精度，`reduce_dtype` 控制梯度通信的精度，`output_dtype` 控制模块输出精度。训练大模型时，常见组合是参数计算用 bf16，梯度 reduce 用 fp32 或按框架策略选择，以平衡显存、带宽和数值稳定性。

这个策略必须原样传给每个 FSDP2 单元。学生实现里如果重新创建 policy，测试里的 identity 检查会失败；生产里则可能出现不同模块 dtype 不一致、通信 dtype 不符合预期、loss 曲线异常或 checkpoint 恢复后 dtype 不匹配。

需要注意，mixed precision 不是“所有地方都降精度”。参数、梯度通信、输出和 optimizer state 可以有不同 dtype。调试时要同时记录 policy、loss、grad norm、overflow 或 NaN 情况，不能只写“用了 bf16”。

**小检查：**

1. `param_dtype` 和 `reduce_dtype` 分别影响哪个阶段？
2. 为什么 patch 要检查 policy 对象是否原样传递？
3. dtype 配错时，你会先看 loss、grad norm，还是只看显存？

## 5. Patch 机制：block-first，root-last

本关的最小接口是：

```python
wrap_transformer_blocks_fsdp2(
    model,
    block_cls,
    mp_policy=None,
    reshard_after_forward=True,
    skip=None,
)
```

实现步骤很窄。先解析 `fully_shard`：测试会注入 spy 函数，生产路径才从 PyTorch 导入。然后遍历 `model.named_modules()`，跳过 root，也跳过不是 `block_cls` 的模块；如果 `skip(name, child)` 返回 True，也不 wrap。对保留下来的 block 按遍历顺序调用 `fully_shard(child, ...)`，最后调用一次 `fully_shard(model, ...)` wrap root。

输出 `WrapReport` 是为了让测试和 drill 有可复查证据。`wrapped_blocks` 记录被 wrap 的 block 名称，`root_wrapped` 说明 root 是否处理，`mp_policy_summary` 把 dtype 策略写成可序列化字典，`reshard_after_forward` 记录配置。真实训练报告里也应该保留这些字段，否则很难解释某次 OOM 或 dtype 问题来自哪里。

**小检查：**

1. 为什么 root 要最后 wrap？
2. `skip` callback 适合排除哪些模块？
3. `WrapReport` 对排查有什么价值？

## 6. TorchTitan 对照：FSDP2 放在并行组合的后段

TorchTitan 的 Llama `parallelize_llama` 不会一上来就 FSDP2。它先检查 sequence length 与 TP/CP 维度是否匹配，再按配置应用 context parallel、tensor parallel、activation checkpoint 和 compile，最后取 data parallel mesh 调用 `apply_fsdp`。这说明 FSDP2 要和其他并行方式一起看，不能孤立调一个 wrap 函数。

`apply_fsdp` 里先创建 `MixedPrecisionPolicy`，再根据配置和 pipeline 状态解析 `reshard_after_forward`。如果模型有 tied weights，embedding、norm、head 会被合在一个 FSDP 单元里处理；否则会分别处理 embedding、输出层相关模块。随后循环 `model.layers.items()`，对每个 transformer block 调用 `fully_shard`，最后 wrap root model。

这一条路径和本关 patch 同构：先确定 policy 和 reshard，再按 block wrap，最后 wrap root。生产源码多出的复杂度来自 device mesh、tied weights、pipeline policy、CPU offload、gradient division 和 checkpoint，而不是另一个完全不同的机制。

**小检查：**

1. TorchTitan 为什么在 FSDP 前处理 TP、CP 和 activation checkpoint？
2. tied weights 为什么可能需要特殊 FSDP 单元？
3. patch reference 和 TorchTitan `apply_fsdp` 的同构点是什么？

## 7. 工程验证：patch-test、dryrun 和 GPU smoke 分别证明什么

`make patch-test M=l12_fsdp2_llama` 主要证明 wrap 行为。CPU 测试用 `_spy_fully_shard` 记录调用顺序，不要求真的初始化 process group，也不证明参数被实际分片。可选 GPU 测试会在 CUDA 和 FSDP2 API 可用时跑 forward/backward 和 state dict 往返，但它仍然是小模型 smoke。

CPU dryrun 通过 `run_fsdp2_smoke.py` 构造一个 Llama-style 小模型，用 spy `fully_shard` 生成 `wrap_report.json` 和 metrics。它适合验证配置、artifact、wrapped block 数量和 report 格式。有 GPU 时，`train_smoke` 会跑 AdamW 训练，记录 loss 变化和 peak memory，才适合讨论 FSDP2 的实际显存边界。

性能判断必须写条件。比如“peak memory 下降”要说明比较对象是 DDP 还是不同 reshard policy，模型规模是多少，micro batch 和 seq length 是多少，硬件和 dtype 是什么。没有这些条件，dryrun 的通过只能说明 wrap 合同成立。

**小检查：**

1. CPU patch-test 和 GPU train smoke 的证据边界有什么区别？
2. `wrap_report.json` 应该包含哪些字段？
3. 讨论 peak memory 时必须记录哪些测试条件？

## 8. 课后思考

1. 如果某个 block 没被 wrap，最可能先表现为显存增加、loss NaN，还是 checkpoint key 缺失？
2. 如果把 `reshard_after_forward` 设成 False，你预期显存和通信会怎样变化？
3. 如果 checkpoint 在不同 world size 恢复失败，你会先检查 FSDP state dict、optimizer state，还是数据集进度？

## 9. 小结

- FSDP2 通过分片参数、梯度和 optimizer state 降低单卡训练显存，但会引入 all-gather、reshard 和 checkpoint 复杂度。
- wrap 粒度决定 FSDP2 的通信和显存生命周期；Llama-style 模型通常先 wrap block，再 wrap root。
- `reshard_after_forward` 是显存与通信之间的开关，不能脱离 pipeline、microbatch 和硬件条件判断。
- `MixedPrecisionPolicy` 要原样传给每个 FSDP 单元，dtype 证据必须进入 report。
- 本关 patch 只验收 wrap 最小合同；真实训练结论要看 GPU smoke、metrics、checkpoint 和 TorchTitan 日志。
