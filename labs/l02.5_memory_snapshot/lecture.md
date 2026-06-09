# L03 讲义：Memory Snapshot 与显存泄露归因

这一讲处理训练系统里很常见的一类事故：训练或 rollout 前几十步都正常，loss、吞吐和 GPU 利用率看起来也没有异常，但显存每隔几步上涨一点，最终在某个 step OOM。这个现象不能只靠 `nvidia-smi` 定位。总量曲线能说明进程占用变大，却不能告诉你哪个 hook、cache、closure 或请求状态把 tensor 留住了。

Memory Snapshot 的思路是把显存问题转成事件和归因问题。每次分配记录 size、时间和调用栈；每次释放把对应分配从 live set 移走；dump 时只看仍然存活的 allocation；最后按调用栈聚合 live bytes，得到最值得先看的 top-k 来源。

本关用 CPU mock 实现这个合同。它不接触真实 CUDA allocator，也不要求 GPU。这样安排的目的很明确：先把事件、live set、caller stack 和聚合这四个概念写对，再迁移到 PyTorch 的 `_record_memory_history` 和 `_dump_snapshot`。

## 1. 本讲目标

学完这一讲，你应该能做到：

1. 区分单步峰值 OOM 和长期泄露型 OOM。
2. 解释为什么 `nvidia-smi`、`memory_summary` 和 snapshot 回答的问题不同。
3. 说清 `AllocEvent`、`_events`、`_live`、`dump_snapshot` 和 stack 聚合的关系。
4. 实现 `MemoryTracker.start/stop/alloc/free/dump_snapshot`。
5. 实现 `get_caller_stack(depth)`，并说明 stack 字符串的用途和噪声。
6. 实现 `find_top_leaks_by_stack(snapshot, k)`，把 live allocations 转成 top-k 候选。
7. 在真实多进程场景中选择正确 rank、worker 和 dump 窗口。

本讲不解决所有 CUDA 内存问题。fragmentation、allocator segment、stream、CUDA graph workspace、NCCL buffer 和真实 snapshot viewer 的细节会在后续课程里继续出现。L03 先训练最小调试心智。

## 2. 从长期 OOM 现象进入

单步峰值 OOM 通常发生得很早。比如 batch size 太大、sequence length 太长、activation 太多，第一步或前几步就会失败。L02 已经讲过这类问题要先看参数、梯度、optimizer state、activation peak 和 profiler。

泄露型 OOM 的时间特征不同。模型能跑起来，每一步只留下少量未释放对象，显存曲线慢慢抬高。到第 80 步、第 200 步或更晚才爆。常见来源包括：

- forward hook 把输入或输出 tensor 存进列表。
- Python closure 捕获了大 tensor，外层引用长期存在。
- rollout cache 或 KV cache 没有按请求完成释放。
- debug logging 把 GPU tensor 放进全局状态。
- 多模态预处理把 image tensor 留在 device 上。

如果只看进程级总量，你能确认“有东西没降下去”，但不能确认来源。Memory Snapshot 要补的证据正是来源：每个仍存活 allocation 来自哪条调用栈。

## 3. 总量指标和 Snapshot 的分工

不同工具回答的问题不同。

| 工具 | 能回答 | 不能回答 |
|---|---|---|
| `nvidia-smi` | 哪个进程占了多少 GPU 显存 | 哪一行代码留下 allocation |
| `torch.cuda.memory_summary()` | allocator 的 reserved/allocated 概况 | 每个 live allocation 的调用栈 |
| profiler | 某段代码的 op、kernel 和时间分布 | 长时间后仍存活对象的来源 |
| Memory Snapshot | alloc/free 历史、live allocations、caller stack | 自动判断业务上是否该释放 |

PyTorch caching allocator 还会制造一个常见误读：reserved memory 不下降，并不直接等于 tensor 泄露。allocator 可能已经释放了 tensor，只是把 block 缓存在进程里供后续复用。泄露分析要看 live allocations 是否随 step 增长，以及这些 live allocations 是否来自同一类 stack。

因此，L03 的判断顺序是：

1. 先看显存是否随 step 单调或阶梯式增长。
2. 再确认增长发生在哪个 rank 或 worker。
3. 开启一段 memory history。
4. 在增长窗口后 dump snapshot。
5. 按 stack 聚合 live allocations。
6. 回到代码修复引用生命周期。

## 4. Snapshot 的最小数据模型

本关的核心结构是 `AllocEvent`：

```python
@dataclass
class AllocEvent:
    addr: int
    size: int
    stack: tuple[str, ...]
    timestamp: float
```

四个字段各有作用。

| 字段 | 含义 | 为什么需要 |
|---|---|---|
| `addr` | 一次分配的唯一标识 | `free(addr)` 需要找到对应 allocation |
| `size` | 分配字节数 | 聚合时计算泄露贡献 |
| `stack` | 调用栈字符串 tuple | 把 live allocation 归因到代码路径 |
| `timestamp` | 分配时间 | 真实排查中可结合 step 和时间窗口 |

本关的 `addr` 是模拟地址。它不是 CUDA 指针，只需要单调变化并能唯一标识 allocation。reference 每次分配后让 `_next_addr += size + 64`，这样连续分配不会拿到相同地址。

## 5. Events 和 Live Allocations

MemoryTracker 维护两本账：

```text
_events: List[("alloc" | "free", AllocEvent)]
_live: Dict[addr, AllocEvent]
```

`_events` 是历史流水，记录发生过什么。`_live` 是当前现场，只保存已经 alloc 且还没有 free 的对象。泄露分析主要看 `_live`，因为已经释放的对象不该计入 `total_leaked_bytes`。

`alloc(size, stack)` 的输入是字节数和调用栈。enabled 时，它生成新地址，创建 `AllocEvent`，把事件写入 `_events`，再把 event 放进 `_live`。输出是地址。

`free(addr)` 的输入是地址。enabled 时，它从 `_live` 中弹出对应 event；如果找到了，就把 `("free", event)` 写入 `_events`。如果地址不存在，安静返回。本关选择 double free safe，是为了让调试工具在复杂清理路径里尽量少制造额外故障。

`dump_snapshot()` 的输出是：

```python
{
    "events": list(self._events),
    "live_allocations": list(self._live.values()),
    "total_leaked_bytes": sum(ev.size for ev in self._live.values()),
}
```

注意 `total_leaked_bytes` 只汇总 live set。把所有历史 alloc 相加会把正常临时 tensor 误报成泄露。

## 6. Caller Stack 为什么是归因核心

只有 size 没有 stack，最多能说“漏了 3GB”。有 stack 才能说“这些 live allocations 多数来自 `forward_hook -> append_image_tensor -> train_step`”。这就是 snapshot 相比总量指标的关键增量。

本关的 `get_caller_stack(depth=4)` 用 `inspect.stack()` 取调用者上溯若干层，并格式化成：

```text
file:line:function
```

这里有两个边界要记住。

第一，Python stack 带噪声。测试、wrapper、trainer loop、decorator 都可能进入 stack。真实排查时经常要按 top frame、归一化文件路径或过滤框架层辅助聚合。本关为了接口简单，按完整 stack tuple 聚合。

第二，stack 只能说明 allocation 创建位置，不能自动说明业务上应该在哪里释放。修复还要回到对象生命周期：hook 是否该 remove，cache 是否该 evict，request finished 是否该 free，closure 是否该改成只保存标量或 CPU copy。

## 7. Stack 聚合如何变成排查候选

单个 live allocation 往往没有信息密度。真正有价值的是同一条调用路径反复留下 allocation。

`find_top_leaks_by_stack(snapshot, k)` 做三步：

1. 遍历 `snapshot["live_allocations"]`。
2. 用 `ev.stack` 作为 key，把 `ev.size` 累加到同一组。
3. 按累计 bytes 降序排序，返回前 `k` 个 `(stack, total_bytes)`。

例如：

```text
stack_a: 1024 + 1024 = 2048 bytes
stack_b: 8192 bytes
```

top-2 应该先返回 `stack_b`，再返回 `stack_a`。测试 `test_find_top_leaks_groups_by_stack` 就是在抓这个语义。

聚合结果不是判决书，它是排查队列。第一名 stack 可能是一个合法长生命周期 cache，也可能是泄露。你需要结合 step、请求生命周期、业务预期和释放路径判断它是否异常。

## 8. Patch 的实现顺序

建议按下面顺序写 starter：

1. 实现 `get_caller_stack`。
2. 实现 `start` 和 `stop`。
3. 实现 disabled 状态下的 `alloc` 返回 `-1`。
4. 实现 enabled 状态下的 `alloc`：生成地址、创建 event、更新 `_events` 和 `_live`。
5. 实现 `free`：跳过 disabled 和负地址，未知地址安静返回。
6. 实现 `dump_snapshot`。
7. 实现 `find_top_leaks_by_stack`。

完成后运行：

```bash
make patch-test M=l02.5_memory_snapshot
```

这 8 个测试都在 CPU 上运行。它们验证的是调试合同，不验证真实 CUDA allocator 性能。

## 9. 真实 PyTorch Snapshot 的边界

真实场景里，PyTorch 的 CUDA memory snapshot 会记录更丰富的信息，例如 allocator segment、block、stream、device、history 和 stack frame。你通常会在泄露窗口前打开 history，在问题复现后 dump 文件，再用 viewer 或脚本分析。

迁移到真实系统时，先问四个问题：

1. 显存上涨发生在哪个进程、rank 或 worker？
2. dump 的窗口是否覆盖了泄露发生的 step？
3. snapshot 记录开销是否会改变训练行为？
4. top stack 对应的是业务 cache、调试引用、hook，还是框架内部长期对象？

多进程 RL 和推理服务尤其要小心。rank0 没有上涨，不代表 rollout worker、SGLang server、vLLM engine 或数据预处理进程没有上涨。dump 错进程会得到一份结构完整但结论无效的 snapshot。

## 10. 生产排查写法

一次合格的 OOM 归因记录至少包含：

- 命令、配置、commit、PyTorch/CUDA 版本。
- 进程、rank、worker、device。
- 显存上涨的 step 区间和采样指标。
- 开启 memory history 的时间点。
- snapshot 路径和 top-k stack 表。
- 判断：top stack 是否符合业务生命周期。
- 修复动作：remove hook、clear cache、释放 request state、保存 CPU copy、缩短引用作用域。
- 验证：同样 workload 下 live bytes 是否不再随 step 增长。

不要只写“snapshot 显示有泄露”。必须把 stack、bytes、时间窗口和修复动作连起来。

## 11. 小结

L03 的核心链路是：

```text
long-running OOM
  -> record alloc/free
  -> keep live allocation set
  -> dump snapshot
  -> group live bytes by stack
  -> inspect top-k owner
  -> fix reference lifetime
```

patch 很小，但它训练的是一条真实排查路径。通过本讲后，你应该能把显存上涨从模糊现象拆成事件、状态、归因和修复验证四步。
