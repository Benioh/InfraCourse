# L37 · CUDA IPC Weight Sync：handle tuple 与共享 storage

这节课拆开 RL 系统里最容易被一句“同步权重”糊过去的机制：训练侧 actor 更新后，怎样把新权重推给 rollout engine。L36 已经练过 dict 形式的语义同步：按 name、shape、dtype 校验后写入推理侧 state。L37 进入真实 co-locate 路径：传输主体是 CUDA IPC handle tuple，不传 tensor data。接收端拿到 handle 和元数据后，可以在自己的进程里重建一个指向同一块 GPU storage 的 tensor。

本关用 CPU 的 `IPCStoragePool` 模拟 CUDA IPC：`serialize_handle` 注册 tensor 并序列化 shape、dtype、stride、device、handle；`deserialize_handle` 按 handle 取回共享 storage；`LocalSerializedTensor` 保存每个 rank 的 handle bytes；`gather_handles_to_rank0` 模拟 rank 0 收齐列表；`update_weights_from_tensor` 根据本 TP rank 替换 inference state，并且只在最后一个参数上清理 pool。学完后，你应该能解释 handle bytes 为什么远小于 tensor bytes，也能判断 `from_tensor`、`from_distributed`、`from_disk` 三条路线的边界。

## 1. 本讲目标

- 解释 CUDA IPC Weight Sync 的 handle tuple 为什么不携带完整 tensor data。
- 画出入口、核心状态、源码主路径和输出 artifact 的关系。
- 说清关键机制的输入、中间状态、输出、代价和边界。
- 读懂本讲 MiniInfra/真实源码中的主函数和关键字段。
- 完成 patch，并用测试、drill 或 notebook 验收最小合同。

## 2. 问题背景和系统位置

本讲属于 `RLHF and rollout systems` 主线。L36 解决的是“哪些 tensor 可以同步”的本地合同；L37 解决的是“co-locate 路径实际传什么”。训练侧 actor 更新后，rollout engine 需要新权重，但同机训练进程和推理进程不一定要复制完整 tensor data。CUDA IPC 路径传 handle tuple 和元数据，让接收端重建指向同一块 GPU storage 的 tensor。

学习时先把系统位置放稳：输入从 workload、配置或请求进入，经过 MiniInfra 或真实框架的核心状态，再落到 metrics、日志、checkpoint、输出文件或服务响应。patch 只检查其中一个最小合同。

## 3. 目标和边界

本讲的目标有四个。第一，能说清 handle tuple 中有哪些信息：shape、stride、dtype、device、CUDA IPC handle、storage offset、refcount 和 event 等元数据。第二，能解释它不包含 tensor data；一个 4 MB fp32 tensor 可以对应几百字节的 bytes。第三，能写出五个接口：`serialize_handle`、`deserialize_handle`、`LocalSerializedTensor.get`、`gather_handles_to_rank0` 和 `update_weights_from_tensor`。第四，能分清 CPU 模拟和真实 CUDA IPC：CPU pool 用 dict 返回同一个 tensor；真实系统用 CUDA IPC 跨进程映射同一块 VRAM。

补丁不启动 Ray、不建 NCCL/Gloo group、不做真实 multiprocessing，也不验证跨进程 CUDA IPC 生命周期。它只验证结构性契约：handle bytes 很小，deserialize 后 `data_ptr()` 一致，rank 0 gather 语义正确，`LocalSerializedTensor` 按 rank 取值，`flush_cache` 只在最后一个 tensor 生效。

**直观理解：**

把大 tensor 想成仓库里的货物，handle tuple 是提货单。weight sync 传提货单，不搬货物；接收端凭提货单找到同一批货物，再把模型参数指向它。

**为什么要学：**

RL 训练一轮可能要同步几十 GB 到上百 GB 权重。若每步都复制数据，rollout 会被同步时间拖死。理解 handle 共享后，才能看懂 co-locate 框架为何能在毫秒到秒级完成更新，也能知道这条路为什么依赖同机共享和进程生命周期管理。

**常见误解：**

- 把 handle bytes 当成 tensor 压缩包；它是元数据和 IPC handle，不携带完整数据。
- 把本关 CPU pool 等同于跨进程 CUDA IPC；CPU 版本只模拟共享 storage 语义。
- 认为 weight sync 只需要 name 对齐；真实路径还要处理 rank、storage、cache flush 和 placement。

**自检问题：**

- 本关需要实现哪五个接口？
- 为什么序列化 bytes 应远小于 tensor 数据字节数？
- CPU 模拟不能证明哪些真实系统结论？

## 4. Handle Tuple 的内容

PyTorch 在跨进程传 CUDA tensor 时，核心在于调用底层 reduction 逻辑产出可重建 tuple，而非把 tensor 数据 pickle 进去。这个 tuple 通常包含重建函数、tensor size、stride、tensor offset、storage 类型、dtype、device、CUDA IPC handle、storage size、storage offset、requires_grad、ref counter handle、event handle 等。接收端根据这些信息重建 Python tensor 对象，并让它映射到同一块 CUDA allocation。

本关的 `serialize_handle` 用更小的 meta dict 模拟这个过程：先 `pool.put(tensor)` 得到 uuid handle，再 pickle `handle`、`shape`、`dtype`、`stride`、`device`。测试会拿 1024x1024 fp32 tensor 做样例，数据约 4 MB；合法 blob 必须小于 1024 bytes。这个断言直接防止学生把 `tensor.numpy()`、`tensor.tolist()` 或原始 storage 塞进 bytes。

`deserialize_handle` 只做两件事：`pickle.loads(blob)` 得到 meta，再 `pool.get(meta["handle"])`。因为 pool 保存的是原 tensor 引用，所以返回 tensor 的 `data_ptr()` 与源 tensor 一致。真实 CUDA IPC 中，进程不同、Python 对象不同，但 storage 指向同一块 VRAM；本关用同对象 data_ptr 相等来模拟这一点。

**直观理解：**

handle tuple 像一张包含仓库地址、货架号、箱子尺寸和访问凭证的单据。它足以让接收方找到货物，单据本身不携带货物。

**为什么要学：**

判断 weight sync 是否高效，第一步就是看传输对象大小。若 handle bytes 接近 tensor bytes，说明实现已经退化成普通复制，co-locate 的主要收益会消失。

**常见误解：**

- 把 `pickle.dumps(tensor)` 和 handle tuple 序列化混在一起；本关只允许序列化 handle 和 meta。
- 只保存 shape/dtype 不保存 handle；接收端无法定位 storage。
- 看到数值相等就认为共享 storage；本关还要求 `data_ptr()` 一致。

**自检问题：**

- handle tuple 中至少需要哪些元数据才能重建 tensor？
- 为什么 4 MB tensor 的合法 blob 应该小于 1 KB？
- 数值相等和共享 storage 有什么区别？

## 5. 从训练侧到 SGLang

co-locate 的 `update_weights_from_tensor` 可以拆成六步。第一，训练侧拿到当前参数；若它是 DTensor 或 FSDP shard，先通过 `full_tensor()` 等方式还原完整 tensor。第二，每个 training TP rank 对完整 tensor 调 `MultiprocessingSerializer.serialize`，得到本 rank 的 handle tuple。第三，所有 rank 参加 `gather_object`，目标 rank 0 收到完整 handle 列表。第四，rank 0 把列表包成 `LocalSerializedTensor(values=[...])`，跨进程传给 SGLang Engine。第五，SGLang 每个 TP rank 调 `LocalSerializedTensor.get(tp_rank)` 反序列化自己对应的 handle。第六，tensor 进入 `ModelRunner.load_weights`，替换推理侧参数。

这里有两个不对称点。`gather_object` 是 collective，所有 rank 都要调用；但只有 dst rank 拿到 list，其它 rank 返回空结果。`LocalSerializedTensor.values` 是 list，因为每个 SGLang TP rank 对应一个来源 rank 的 handle blob；如果只传单个 blob，所有 TP rank 会拿到同一份 storage，无法表达不同 rank 的切片或转换结果。

SGLang 侧的 `_unwrap_tensor` 会先处理 `LocalSerializedTensor`，再把 tensor 移到当前 device。真实代码会调用 `monkey_patch_torch_reductions`，让 PyTorch CUDA tensor reduction 中的 device 信息可跨进程/跨 rank 正确解析。本关 CPU 实现不需要 monkey patch，但要保留“按 rank get，再更新 state”的结构。

**直观理解：**

训练侧每个 rank 各自开一张提货单，rank 0 把所有提货单装进信封交给 SGLang。SGLang 的每个 TP rank 只抽取属于自己的那张。

**为什么要学：**

weight sync 失败常见于 rank 维度错配：所有 rank 没一起 gather、rank 0 没收齐、LST values 顺序错、推理 TP rank 取错 handle。拆开链路能把这些问题定位到具体一步。

**常见误解：**

- 认为只有 rank 0 调 gather；collective 需要所有 rank 参与。
- 把 `values` 设计成单个 bytes；真实接口需要按 TP rank 保存多个 handle blob。
- 忽略 values 的顺序；`values[rank]` 与 TP rank 的对应关系必须稳定。

**自检问题：**

- 为什么 `gather_object` 后只有 rank 0 拿到完整列表？
- `LocalSerializedTensor.values` 为什么是 list？
- SGLang 每个 TP rank 在哪里选择自己的 handle？

## 6. 实现 Gather 和 LST

`gather_handles_to_rank0` 是本关对 `dist.gather_object` 的 CPU 模拟。函数每次都先写 `group[rank] = local_blob`。如果 `rank != 0`，直接返回 `None`；如果 `rank == 0` 且 `len(group) < world_size`，也返回 `None`；只有 rank 0 看到所有 rank 都写入后，才返回 `[group[0], group[1], ..., group[world_size-1]]`。测试特意让 rank 0 先调用一次，再让其它 rank 写入，最后 rank 0 再轮询一次拿到完整列表。

`LocalSerializedTensor.get(rank, pool)` 很薄，只是调用 `deserialize_handle(self.values[rank], pool)`。薄并不代表它不重要：它把“一个参数在每个 TP rank 上有不同 handle”这个结构明确下来。后续 `update_weights_from_tensor` 不需要知道 gather 细节，只要按本 rank 调 `lst.get(tp_rank, pool)`。

写实现时不要让非 0 rank 返回自己的 blob，也不要让 rank 0 在没收齐时返回部分列表。部分列表会让后续 rank 索引越界或拿到旧 blob。也不要在 `LocalSerializedTensor.get` 里 clone；本关要求返回共享 storage 的 tensor，clone 会让 `data_ptr()` 变化。

**直观理解：**

`group` 是一个临时信箱。每个 rank 投递自己的信，rank 0 只有在所有信都到齐后才把整叠信取走。

**为什么要学：**

分布式 bug 往往出在 collective 语义，而非公式本身。用 CPU dict 复现 gather 的不对称性，可以在没卡环境中先练清楚 rank 0 的职责和非 0 rank 的返回值。

**常见误解：**

- 让 rank 0 第一次调用就返回列表；此时其它 rank 可能还没写入。
- 让所有 rank 都拿完整列表；这偏离 gather_object 的单目标接收语义。
- 在 LST.get 中返回 clone；这样会破坏共享 storage 检查。

**自检问题：**

- rank 0 在 group 未收齐时应该返回什么？
- 非 0 rank 调用 gather 后应该得到什么？
- 为什么 LST.get 不应该 clone tensor？

## 7. 替换权重和 Flush 时序

`update_weights_from_tensor` 接收 `(name, LocalSerializedTensor)` 列表、推理侧 state、当前 `tp_rank`、pool 和 `flush_cache`。循环中每个参数都按 `lst.get(tp_rank, pool)` 取出本 rank 的 tensor，再写到 `inference_state[name]`。在本关 CPU state 里，这相当于把旧 tensor 引用替换成共享 storage 的新 tensor；真实 SGLang 会把 tensor 交给 `model.load_weights` 或特定 loader。

`flush_cache` 的时序很关键。权重变了以后，旧 KV cache 和 radix tree 中的缓存结果不再可靠；但一次 weight update 往往包含很多参数。如果每更新一个 tensor 就 flush，prefix/radix cache 会反复清空和重建。任务要求只在 `flush_cache=True` 且 `tensor_index == len(named_handles)-1` 时清理 pool，用这个条件模拟“整批参数更新完成后再统一清 cache”。

测试用两个 tensor 验证这个行为：`flush_cache=False` 时 pool 保留两个 handle；`flush_cache=True` 时循环跑到最后一个参数才 `pool.clear()`，最终 pool size 为 0。这个测试不关心真实 SGLang 的 radix tree 数据结构，但它能防止学生把 cache 清理写在循环开头或每个参数后。

**直观理解：**

替换权重像给机器逐个换零件。缓存清理应等一整套零件换完再做；中途反复清理只会浪费时间，也可能让状态边界更难推理。

**为什么要学：**

weight sync 的正确性不止是参数数值对。生成服务还有 KV cache、prefix cache 和正在处理的请求；flush 过早或过晚都会影响吞吐或正确性。

**常见误解：**

- 把 pool.clear 写在循环外无条件执行；`flush_cache=False` 时不应清理。
- 每个参数更新后都清 pool；任务要求最后一个参数才触发。
- 只看参数替换，忘记旧 KV cache 与旧权重绑定。

**自检问题：**

- `flush_cache=True` 时应在循环的哪个位置清 pool？
- 为什么旧 KV cache 在权重更新后需要失效？
- 本关的 `inference_state[name] = tensor` 在真实 SGLang 中对应哪类操作？

## 8. Slime 的分桶更新

slime 的 co-locate 路径也使用 handle 传递，但它面对 Megatron、PP/EP/TP 多维并行和大 MoE 模型，不能把整份模型权重常驻 GPU 后一次性同步。它会先把参数按 `update_weight_buffer_size` 切成 bucket；每次只 upload 一个 bucket 的 Megatron 权重到 GPU，在 PP/EP 维度 broadcast，在 TP 维度 all_gather/cat 成完整 tensor，再转换为 HF 命名格式并序列化。

发送给 colocated engine 时，slime 会把一个 bucket flatten 成 `FlattenedTensorBucket`，把 flattened tensor 和 metadata 一起序列化，在 `_ipc_gather_group` 中 gather 到源 rank，再调用 rollout engine 的 `update_weights_from_tensor`。这些 long-lived tensors 要在消费者关闭 IPC handle 后再释放，所以代码里有 `torch.cuda.ipc_collect()` 和 barrier。

这套设计主要控制显存峰值。训练侧权重、SGLang model weights、KV cache、CUDA Graph pool 不能随意叠加。分桶让临时上传的训练权重保持在较小窗口内，代价是同步逻辑更复杂，还要维护 metadata、weight version、pause/continue generation 和 cache flush。

**直观理解：**

verl 路径像一件一件参数寄提货单，slime 路径像按箱打包：每箱有一块 flattened tensor 和一份清单，收件端按清单拆回多个参数。

**为什么要学：**

大 MoE 或多维并行训练里，weight sync 的瓶颈常常是峰值显存和转换路径，而不只是 handle 大小。理解 slime 分桶后，学生能把“为什么同步慢”拆成 upload、broadcast、gather、serialize、Ray IPC、load_weights 和 cache flush。

**常见误解：**

- 以为 from_tensor 就完全没有显存峰值；聚合完整 tensor 和 bucket 临时上传仍会占显存。
- 只看 IPC handle 很小，忽略 PP/EP/TP 聚合和格式转换也要时间。
- 把 bucket 当成性能装饰；它主要是控制大模型同步时的显存窗口。

**自检问题：**

- slime 为什么要按 bucket 更新权重？
- PP/EP/TP 聚合分别解决什么切分问题？
- 为什么发送后还需要 `torch.cuda.ipc_collect()`？

## 9. 三种同步接口取舍

`update_weights_from_tensor` 适合 co-locate：训练进程和推理进程在同一节点或能共享 IPC 的环境里，传 handle metadata，数据通过共享 VRAM 暴露给接收端。它传输量小、同步快；限制是 placement 侵入性强，资源布局要配合，动态扩缩容不方便。

`update_weights_from_distributed` 适合 disaggregate：训练资源组和推理资源组分开常驻，通过 NCCL 或 IB 传 tensor 数据本身。它支持更灵活的服务集群，但要建立跨组通信，传输量与权重大小相关，扩缩容也要处理通信组变化。

`update_weights_from_disk` 最简单：训练侧写 checkpoint 或权重文件，推理侧从磁盘加载。它 I/O 开销大，但工程边界清晰，checkpoint 管理顺手，rollout 动态扩缩容更容易。选择哪条路要看 placement、模型大小、同步频率、容错需求和集群弹性。

**直观理解：**

from_tensor 是递提货单，from_distributed 是走专线搬货，from_disk 是放到仓库再让对方自取。三条路的快慢和适用场景不同。

**为什么要学：**

RL 系统设计没有单一同步答案。co-locate 追求低延迟，但牺牲一部分服务弹性；disaggregate 更利于独立扩容，但同步链路更重；磁盘方案慢一些，却常常最容易恢复和扩展。

**常见误解：**

- 把 from_tensor 当成所有 placement 的默认选项；它依赖 co-locate 和共享内存条件。
- 认为 from_disk 一定不可用；它慢，但扩缩容和 checkpoint 管理简单。
- 把 from_distributed 和 from_tensor 都叫零拷贝；前者传数据，后者传 handle metadata。

**自检问题：**

- from_tensor 的主要前提是什么？
- 为什么 disaggregate 场景更常考虑 from_distributed？
- 在哪些需求下 from_disk 反而更容易落地？

## 10. 测试、报告和硬件证据

本关 7 个 pytest 对应 7 个行为契约：handle bytes 小于 1 KB；deserialize 后 data_ptr 一致；round trip 数值相等；非 0 rank gather 返回 None，rank 0 收齐列表；LST 按 rank get；update 后 inference_state 被替换；flush_cache 只在最后一个 tensor 清 pool。它们足以检查 CPU patch 语义。

这些测试不能替代真 CUDA IPC 验证。真实硬件上还要看跨进程 spawn 是否能传 CUDA IPC handle，接收端 tensor 是否指向同一块 VRAM，源 tensor 生命周期是否覆盖接收端使用窗口，`ipc_collect` 是否及时释放，weight sync 的 wall time 和 handle bytes 是否符合预期。

报告建议记录五类证据：每个 tensor 的 data bytes、handle bytes、handle/data 比例；pool 或 IPC handle 数量；rank0 gather 是否收齐 world_size；每个参数 update 耗时和总 sync time；flush_cache 的调用次数和位置。若进入 GPU 集群，再补充 CUDA IPC 跨进程脚本、NCCL/Gloo group 配置、cache flush 日志和 rollout weight_version。

**直观理解：**

CPU 测试证明“提货单格式对、能取到同一批货”；GPU 测试证明“跨进程仓库真的能按提货单开门，而且用完会关门”。

**为什么要学：**

weight sync 出问题时，reward 曲线通常只会滞后地表现异常。提前把 handle 大小、共享 storage、rank gather、cache flush 和版本证据写进日志，能把同步 bug 从训练效果里拆出来。

**常见误解：**

- 只跑 CPU pytest 就声称 CUDA IPC 跨进程可用；硬件路径还要单独验证。
- 只看 handle bytes，不检查 data_ptr 或 storage 生命周期。
- 不记录 weight_version 和 flush 位置；后续很难判断 rollout 是否用了新权重。

**自检问题：**

- 本关 7 个 pytest 分别覆盖哪些契约？
- 真 CUDA IPC 验证还需要补哪些证据？
- 一份 weight sync 报告至少应记录哪些大小、rank 和时序指标？

## Lab 验收边界

本讲 patch 命令：`make patch-test M=l32.5_ipc_weight_sync`。

patch 验收的是：实现 verl/slime co-locate 路径下 update_weights_from_tensor 的核心机制：序列化传 IPC handle 而非 tensor data；接收端反序列化后与源 tensor 共享 storage。CPU 模拟通过 IPCStoragePool 实现，API 形状与真实代码一致。

这不是整节课的全部。测试通过后，还要能把测试中的输入、状态变化、异常边界和真实源码主路径对上。

## 生产排查入口

遇到同类问题时，按下面顺序排查：

1. 先确认 workload、配置、版本、硬件和随机种子。
2. 再看入口日志、核心状态、指标和 artifact 是否完整。
3. 然后沿源码主路径定位状态变化发生在哪一层。
4. 最后比较 patch 或 MiniInfra 的最小合同，判断真实系统多出的复杂度来自哪里。

课后使用 `outputs/rl_rollout_template.md` 记录一次完整复盘。
