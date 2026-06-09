# L34：CUDA Graph Cache + Memory Savor

RL co-locate 的目标是在同一批 GPU 上交替运行 rollout 和 training。rollout 侧需要高吞吐 decode，training 侧需要权重、optimizer state 和 activation。两边同时常驻容易 OOM；直接释放 rollout 显存又可能破坏 CUDA Graph replay 所依赖的地址稳定。L34 就围绕这个矛盾展开：怎么让 decode 少受 CPU launch overhead 影响，同时让训练阶段拿到足够显存。

本讲 patch 是 CPU 语义模拟，不会调用 `torch.cuda.graph`。你要实现两个结构：`GraphCache` 按输入 shape/dtype/scalar 生成 key，首次见到 key 记录函数并计 capture，后续命中计 replay；`MemorySavor` 把 tensor 数据保存到 handle 池，pause 后统计 bytes，resume 后恢复数据并删除 handle。学完后，你应该能把 CPU 结构语义和真实 GPU 证据分开。

## 1. 本讲目标

- 解释 CUDA Graph replay 减少的是哪类开销。
- 说明真实 replay 为什么依赖 static buffer 和稳定地址。
- 解释 Memory Savor 在 co-locate 阶段切换中的作用。
- 写出 `GraphCache` 和 `MemorySavor` 的最小 CPU 行为合同。
- 读懂 Megatron CUDA Graph metadata 与 SLiME memory utility 的相关源码。

## 2. CUDA Graph 解决什么开销

PyTorch eager 执行中，每个 GPU kernel 都需要 CPU 发起 launch。decode 阶段一次只生成一个 token，单步通常包含很多小 kernel：layer norm、attention 片段、采样前后的张量操作、小矩阵乘。单个 kernel 计算越短，CPU launch overhead 在总延迟中的占比越高。

CUDA Graph 的思路是把一段固定 GPU 操作序列录制成可 replay 的执行对象。replay 时 CPU 只提交一次 graph launch，GPU 按已录好的依赖关系执行整段 kernel 序列。它适合 shape 稳定、控制流稳定、内存分配稳定的路径，典型位置是推理 decode。prefill 的 prompt 长度和 batch 组成更动态，graph cache 命中率要单独评估。

这里要避免一个误解：CUDA Graph 主要减少 kernel launch 的 CPU 提交开销，不会让大 GEMM 本身神奇变快。收益大小取决于 kernel 粒度、batch size、并发和 graph 命中率。

## 3. 地址稳定与 static buffer

真实 CUDA Graph replay 的硬约束是地址稳定。capture 阶段会记录 kernel 参数和 tensor 对应的 GPU 地址。graph 实例化之后，replay 会继续读写这些地址。若下一次请求换成新分配 tensor，即使 shape 和 dtype 相同，地址也可能变，graph 就可能读写错误位置。

生产 graph runner 通常会预分配 static input/output buffer。每次 replay 前把新输入 `copy_` 到同一块 input buffer，graph 内部读取固定地址；输出也写入固定 output buffer。这样动态数据可以进入 graph，但进入方式必须是不改变地址的就地写入。

batch size 也会影响 graph。`bs=4` 和 `bs=8` 可能对应不同 grid、临时 tensor shape、KV cache 索引和内存布局，所以服务端常按 batch bucket capture 多个 graph。bucket 太少会 miss，bucket 太多会占显存和初始化时间。

## 4. GraphCache 的 CPU 合同

本关 `GraphCache` 不录 CUDA kernel。它训练的是控制面：怎么判断一次调用是 capture 还是 replay。`_shape_key` 对 tensor 返回 `("tensor", shape, dtype)`，对 scalar 返回 `("scalar", type_name, value)`。`capture_or_replay(fn, *args, **kwargs)` 把 args 和排序后的 kwargs 转成 key。

首次遇到 key 时，`GraphCache` 保存 `fn` 引用，`capture_count += 1`，并执行这次函数。再次遇到相同 key 时，`replay_count += 1`，调用缓存里的函数。测试里第二次传入新的 lambda 也会命中旧函数，这模拟真实 graph 录完后不会跟随 Python 代码变化。

这个 CPU 版本不验证 `data_ptr()`，也没有 static input buffer。它验证的是 shape-key cache、capture/replay 计数、不同 shape 触发新 capture、replay 输出与 eager 一致。报告中不能把这些结果写成 GPU 加速结论。

## 5. MemorySavor 的 pause/resume

Memory Savor 的真实动机来自 CUDA Virtual Memory。普通 `cudaFree` 会释放显存并使虚拟地址失效；captured graph、KV cache pool 或其他持有该地址的对象都会变危险。VMM 风格的做法是保留虚拟地址范围和元数据，pause 时解除物理页映射并释放物理内存，resume 时重新映射物理页回同一段虚拟地址。

CPU patch 用字典模拟这个状态机。`pause(tensor)` 生成递增 handle，把 shape、dtype 和 `tensor.detach().clone().cpu()` 存入 `_paused`，返回 handle。`resume(handle)` 从 `_paused` 弹出元数据并返回数据 clone。`total_paused_bytes()` 按 `numel * element_size` 统计暂停池里的数据规模，`is_paused(handle)` 判断 handle 是否仍在池中。

这里的关键在状态机，clone 只是 CPU 模拟的存储方式：pause 后有可追踪 handle，resume 只能消费一次，bytes 统计随 handle 增减变化。真实系统还需要按 region/tag 管理 rollout KV cache、weights 或 allocator pool；本讲先练最小 handle 合同。

## 6. co-locate 的切换顺序

一个典型 co-locate 切换顺序是：rollout 完成后，pause rollout 侧大块显存；训练侧上传或恢复权重，执行 forward/backward/update；训练结束后 offload training 权重和临时状态；最后 resume rollout 侧显存并继续采样。

顺序错了会直接影响峰值显存。若 rollout 还没 pause 就恢复 training weights，KV cache、推理权重、训练权重和 activation 可能同时存在。若直接销毁 rollout 进程或释放 buffers，下一轮恢复成本高，graph 内部地址也可能失效。若 training offload 前就 resume rollout，两套大对象会再次重叠。

GraphCache 和 MemorySavor 的交集是地址。GraphCache 希望 decode 的 static buffer 和 captured graph 保持稳定；MemorySavor 希望训练阶段让出物理显存，同时保留恢复所需的元数据和地址语义。真实系统能否兼顾这两点，需要硬件证据。

## 7. 真实源码中的信号

Megatron 的 CUDA Graph 代码记录了 capture 状态、warmup 状态、tensor metadata 和 buffer reuse pool。`ArgMetadata` 会保存 tensor 的 shape、dtype、device 和 `data_ptr()`；`TensorReusePool` 会保存强引用和 data pointer 集合，防止 buffer 被 allocator 回收。这些细节都指向同一个约束：graph capture 不是只看 shape，还要管理地址和 buffer 生命周期。

Megatron 的参数里也能看到生产边界：动态图推理可以配置要 capture 的 CUDA Graph 数量，也有 decode-only graph 选项；RL 相关参数还区分 KV cache persist/offload/recompute，以及 inference engine suspend 时是否保留 CUDA Graph。SLiME 的 memory utility 则提供了清理 cache、查询 GPU/host memory 和按 rank 打印内存信息的基础工具。

本讲只把这些生产复杂度映射到最小 patch。读源码时不要把 patch 当真实替代品；它是理解 shape key、handle 和 bytes 统计的入口。

## 8. 测试和报告边界

pytest 覆盖 8 个行为：首次调用 capture、第二次同 shape replay、不同 shape 新 capture、replay 输出与 eager 一致、pause 后 handle 在池中、resume 返回相同数据、paused bytes 统计正确、resume 后从池中移除并让 bytes 归零。

这些测试不能证明 CUDA Graph 加速，也不能证明 GPU 地址稳定。真正的 GPU 验证至少要记录：static input/output buffer 的 `data_ptr()` 是否跨 replay 不变，graph replay 时间是否低于 eager，多 bucket graph 的命中率和显存高水位，pause/resume 后 captured graph 是否还能 replay。

一份合格报告应分三栏：CPU 已验证、GPU 待验证、生产风险。CPU 已验证写 patch-test 和 notebook 现象；GPU 待验证写地址、时间和显存；生产风险写动态 shape、host-device sync、初始化顺序、bucket 配置和 co-locate 切换日志。

## Lab 验收边界

本讲 patch 命令：`make patch-test M=l30.5_cuda_graph_savor`。

patch 验收的是 `GraphCache` 与 `MemorySavor` 的 CPU 行为合同。它不覆盖真实 CUDA Graph capture/replay、CUDA stream、VMM API、真实物理显存释放或多进程 co-locate。

课后使用 `outputs/rl_rollout_template.md` 记录一次完整复盘。

---

## 补充：CUDA Graph 加速原理

### 1. 为什么 CUDA Graph 有效

常规 PyTorch 执行中，每个 CUDA kernel 的启动路径为：

```
Python 层 dispatch → C++ ATen dispatch → cudaLaunchKernel API → GPU 执行
```

每次 kernel launch 的 host 端开销约 **5-20μs**（包括参数校验、stream 同步点检查、driver 调用）。对于大 kernel（如 GEMM 执行数 ms），这个开销可忽略；但对于小 kernel（如 elementwise、layernorm、activation，执行 10-50μs），launch overhead 可占总时间的 20-50%。

CUDA Graph 的做法：**录制（capture）一段 kernel 序列**，形成一个 graph 对象，之后**重放（replay）时只需一次 API 调用**即可提交整个序列。host 端开销从 `N × launch_cost` 降到 `1 × graph_launch_cost`。

### 2. 消除的具体开销

CUDA Graph replay 绕过的开销包括：

- **Python dispatch**：`torch.xxx()` 的 Python 函数调用、参数解析、dtype 推导
- **C++ dispatch**：ATen 的 dispatcher（找到正确的 kernel 实现）
- **CUDA driver 调用**：`cudaLaunchKernel` 的参数打包、stream 排队
- **内存分配器决策**：每个中间 tensor 的 `cudaMalloc`/caching allocator 查找

总计每个 kernel 节省 5-20μs。如果一个 decode step 有 200 个小 kernel，总共节省 1-4ms——对于 decode 阶段（目标 ITL < 10ms）这是显著收益。

### 3. 静态地址要求

CUDA Graph 的核心约束：**capture 和 replay 时所有 tensor 的 GPU 地址必须相同**。

原因：graph 录制时把每个 kernel 的参数（包括输入/输出 tensor 的 `data_ptr()`）固化到 graph 节点中。replay 时直接使用这些地址，不做任何检查。

实践影响：

- 输入 tensor 必须是 **static buffer**：预分配固定大小，每次 replay 前把新数据 copy 到同一块 buffer。
- 中间 tensor 由 CUDA memory pool 在 capture 时分配，replay 时复用相同地址。
- **不能在 capture 期间做动态分配**（如 `torch.empty` 的 size 依赖 runtime 值）。
- Shape 变化需要重新 capture 一个新 graph（或维护多个 graph 对应不同 shape）。

### 4. CUDA Graph 收益最大的场景

- **大量小 kernel**：decode 阶段的 attention（多个小 GEMM + softmax + mask）、layernorm、residual add。
- **Inference / Decode 阶段**：batch 内 token 数少，每个 kernel 计算量小，launch overhead 占比高。
- **固定 shape 推理**：输入 shape 不变（如固定 batch_size 的 decode），一个 graph 可反复 replay。
- **延迟敏感场景**：实时 serving 要求 ITL < 10ms，每 ms 的节省都重要。

### 5. CUDA Graph 收益有限的场景

- **大 kernel 主导**：prefill 阶段的大 GEMM（执行数 ms），launch overhead 占比 < 1%，graph 收益可忽略。
- **动态 shape**：每次输入 shape 不同（如 variable-length prefill），无法复用 graph，反而增加 capture 开销。
- **复杂控制流**：模型中有 if/else、dynamic routing（如 MoE 的 top-k 选择），无法静态录制。
- **已经被 compute 打满**：GPU SM 利用率已经接近 100%，瓶颈不在 launch overhead。

### 6. 与 torch.compile 的集成

`torch.compile` 可以自动利用 CUDA Graph：

- **mode="reduce-overhead"**：compile 会自动 capture CUDA Graph。它分析计算图，找到静态子图（shape 不变的部分），对这些子图做 graph capture。
- **自动 padding**：对多种 input shape，compile 可能 pad 到最近的 bucket size 后共用一个 graph。
- **与 custom kernel 兼容**：triton kernel 也可以被纳入 graph capture。

```python
model = torch.compile(model, mode="reduce-overhead")  # 自动使用 CUDA Graph
```

手动 capture 仍然有用的场景：需要精确控制 capture 边界、需要 graph 池管理（如 vLLM 的多 bucket graph cache）。

### 7. 显存影响

CUDA Graph 对显存有额外占用：

- **Captured graph 持有所有中间 tensor 的引用**：capture 期间分配的中间 tensor 不会被释放，因为 replay 要复用它们。一个 graph 的显存占用 = 所有中间激活 tensor 的总大小。
- **多个 graph = 多份中间 tensor**：如果为不同 batch_size/seq_len 各 capture 一个 graph，显存占用线性增长。
- **Static input/output buffer**：额外预分配的固定大小 buffer。

显存管理策略：

- 限制 graph 池大小（如最多 cache 8 个 graph）。
- 对不常用的 graph 做 eviction（释放其 tensor，下次重新 capture）。
- 使用 CUDA VMM API（`cudaMallocAsync`）让多个 graph 共享物理显存池。
- vLLM 的 `GraphCache` 就是在管理这个 tradeoff：更多 cached graph = 更低延迟但更高显存。

总结：CUDA Graph 是一种用显存换延迟的技术——预付显存成本（持有中间 tensor），换取 replay 时零 launch overhead。适合 decode 阶段等小 kernel 密集、shape 固定、延迟敏感的场景。
