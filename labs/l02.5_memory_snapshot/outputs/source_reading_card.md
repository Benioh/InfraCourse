# L03 Source Reading Card

这张卡用于快速复习 Memory Snapshot patch 的源码主路径。读源码时按顺序走，先抓状态，再看测试。

## 1. 主路径

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `labs/l02.5_memory_snapshot/patch/starter/memory_snapshot.py` | `AllocEvent`、`get_caller_stack`、`MemoryTracker` TODO、聚合 TODO | 学生要补齐的是事件记录、live set 和 stack 聚合 |
| `labs/l02.5_memory_snapshot/patch/reference/memory_snapshot.py` | `alloc`、`free`、`dump_snapshot`、`find_top_leaks_by_stack` | 状态变化顺序决定 snapshot 是否可信 |
| `labs/l02.5_memory_snapshot/patch/tests/test_patch.py` | 8 个测试函数 | 测试覆盖 disabled、平衡释放、泄露、double free、shape、聚合、top-k 和 caller stack |

## 2. 最小不变量

- disabled tracker 的 `alloc` 返回 `-1`，不能改 `_events` 或 `_live`。
- enabled `alloc` 必须生成新 addr，写入 `_events` 和 `_live`。
- `free` 只移除 live allocation，未知 addr 安静返回。
- `dump_snapshot` 的泄露总量只来自 live allocations。
- `find_top_leaks_by_stack` 按完整 stack tuple 聚合 size，并按 bytes 降序返回。
- `get_caller_stack(depth)` 返回 tuple of string，格式为 `file:line:function`。

## 3. 读源码时的提问顺序

1. 这个函数读写 `_events`、`_live`、`_enabled` 还是 `_next_addr`？
2. 它的输入是什么，输出给谁消费？
3. disabled、unknown addr、double free 这类边界会怎样处理？
4. snapshot 中哪部分是历史，哪部分代表当前现场？
5. 测试断言覆盖了哪条行为合同？

## 4. 迁移到真实系统时要补的信息

- rank / worker / pid / device。
- step 或请求生命周期。
- PyTorch/CUDA 版本和 allocator 配置。
- snapshot dump 的时间窗口。
- top stack 是否属于业务代码、hook、cache 或框架内部长期对象。
