# L03 源码带读：MemoryTracker 的事件流

这份带读按“接口合同到测试验收”的顺序走。读源码时不要从文件顶部一路滚到底，直接按下面的锚点看。

建议先读 [lecture.md](lecture.md)，确认 events、live allocations 和 stack aggregation 的关系，再打开源码。

## 0. 源码地图

```text
labs/l02.5_memory_snapshot/patch/starter/memory_snapshot.py
  -> 学生要补齐的 TODO 和接口边界

labs/l02.5_memory_snapshot/patch/reference/memory_snapshot.py
  -> 最小可通过实现，展示状态变化顺序

labs/l02.5_memory_snapshot/patch/tests/test_patch.py
  -> 8 个 CPU tests，锁定行为合同和边界
```

`Makefile` 只是命令入口，不放进源码阅读主线。真正需要精读的是 Python 文件里的状态和测试断言。

## 1. Starter：先识别要维护的状态

文件：[memory_snapshot.py](patch/starter/memory_snapshot.py)

先看第 21-27 行：

```python
@dataclass
class AllocEvent:
    addr: int
    size: int
    stack: Tuple[str, ...]
    timestamp: float
```

结论：一次 allocation 的最小证据包含地址、大小、调用栈和时间。后续 `_events`、`_live` 和聚合函数都围绕这个结构。

再看第 29-37 行。`get_caller_stack` 的 TODO 已经给出实现提示：跳过本函数自身，从调用者开始取 `depth` 层，并格式化成 `file:line:function`。读完后要能解释为什么 stack 必须是 tuple of string，而不是临时 frame 对象。

看第 40-47 行。`MemoryTracker` 初始化四个状态：

- `_events`：历史 alloc/free 事件。
- `_live`：当前仍存活的 allocation。
- `_enabled`：是否记录。
- `_next_addr`：模拟地址生成器。

这四个状态就是本关 patch 的全部数据面。

## 2. Reference：按状态变化读实现

文件：[memory_snapshot.py](patch/reference/memory_snapshot.py)

先看第 31-35 行。`start` 和 `stop` 只改 `_enabled`。它们不清空 `_events` 或 `_live`，因此一次录制窗口内已有的证据不会被误删。

再看第 37-45 行。`alloc` 的顺序不能乱：

1. disabled 时返回 `-1`。
2. 取当前 `_next_addr`。
3. 推进 `_next_addr`，保证后续地址不同。
4. 创建 `AllocEvent`。
5. 写 `_events`。
6. 写 `_live`。
7. 返回 addr。

看第 47-52 行。`free` 只在 enabled 且地址非负时处理。未知地址通过 `pop(addr, None)` 安静返回；找到 event 时才记录 free 事件。

看第 54-59 行。`dump_snapshot` 返回三项：完整事件、当前 live allocations 和 live bytes 总和。它没有把历史 alloc 总量当成泄露。

最后看第 62-69 行。聚合函数只读 snapshot，不改 tracker 状态。它按 `ev.stack` 分组、累加 size、降序排序、截断 top-k。

## 3. Tests：用断言反推合同

文件：[test_patch.py](patch/tests/test_patch.py)

第 21-27 行验证 disabled tracker。没有 `start()` 时，`alloc` 必须返回 `-1`，snapshot 总泄露为 0。这能防止录制窗口外的行为污染证据。

第 30-39 行验证 alloc/free 平衡。分配后释放，live allocations 应为空，`total_leaked_bytes` 为 0。

第 42-50 行验证未释放 allocation。只 alloc 不 free 时，snapshot 里应该出现 2048 bytes 的 live allocation。

第 53-62 行验证 double free safe。重复释放和释放未知地址都不能抛错，最终也不能留下泄露。

第 65-78 行验证 snapshot shape。测试检查 `events`、`live_allocations`、`total_leaked_bytes` 三个 key，并确认两个 alloc event 被记录。

第 81-97 行验证 stack 聚合。两个 `stack_a` allocation 要合成 2048 bytes，`stack_b` 的 8192 bytes 应排第一。

第 100-111 行验证 top-k 截断。5 个不同 stack 只取前 2 个，并按 size 降序。

第 114-120 行验证 caller stack 格式。返回值必须是 tuple，元素必须是 string，第一帧应包含当前测试函数名。

## 4. 可以先跳过的内容

本关源码很小，但读真实 PyTorch snapshot 时会遇到很多生产分支。第一次阅读可以先跳过：

- allocator segment、block split/merge 和 stream 细节。
- snapshot viewer 的 UI 展示逻辑。
- 多 GPU 进程间收集和文件命名策略。
- 按 frame 归一化、过滤框架栈和符号化路径的高级分析。

这些分支会影响真实工具体验，但不会改变 L03 的主合同：记录事件，维护 live set，按 stack 聚合。

## 5. 读完后的自检问题

1. `AllocEvent` 的四个字段分别服务于哪类排查问题？
2. `_events` 和 `_live` 的区别是什么？
3. disabled tracker 为什么不能改内部状态？
4. double free 为什么要安静返回？
5. `dump_snapshot` 为什么只把 live allocations 计入泄露总量？
6. `find_top_leaks_by_stack` 的排序比较对象是什么？
7. 真实 CUDA snapshot 相比本关 mock 多了哪些信息？
