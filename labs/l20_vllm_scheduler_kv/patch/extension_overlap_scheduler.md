# L20 扩展 · Zero-Overhead Overlap Scheduler + FutureMap

> 这是 L20 的可选扩展挑战。**不在 patch-test 范围内**，不会影响主线测试通过。
> 完成后你会理解 SGLang 在 v0.4 后最重要的调度优化（怜悯 24/12 的工作）。

## 背景

L20 主线实现的是 normal scheduler：CPU pre/post schedule 与 GPU compute/sample 串行。
profiling 一下你会发现，CPU pre/post schedule 占了大约 30%–50% 的总耗时——
GPU 在 idle 等 CPU 排序、组装张量、detokenize、判 EOS。

SGLang 的解法是 **overlap scheduling**：用 `FutureMap` 让 CPU 在不知道真实 token 值的情况下
就开始组装下一轮的 input。GPU 的 sample 阶段写真实值进 FutureMap，下一轮的 resolve kernel
按索引读出来。

## 你要扩展什么

在 `patch/starter/scheduler.py` 之外，新加一个文件 `patch/starter/scheduler_overlap.py`：

```python
class FutureMap:
    """GPU 端的"占位符 → 真实 token id"映射表。"""
    def reserve(self, batch_size: int) -> list[int]:  # 返回 indices
        ...
    def write(self, indices: list[int], values: list[int]) -> None:
        ...
    def read(self, indices: list[int]) -> list[int]:
        ...

class OverlapScheduler(Scheduler):
    """与 normal Scheduler 接口兼容，但 schedule() 的输出包含 future_indices；
    GPU 在 sample 后调 future_map.write 填值，下一轮 schedule 通过 read 拿到。"""
    def event_loop_overlap(self) -> Iterable[SchedulerOutput]:
        # 1. result_queue = deque()
        # 2. while True:
        #    a. cur_batch = self.schedule()  # 用上一轮的 future_indices 作为输入
        #    b. fake_input_ids = [FUTURE_PLACEHOLDER] * len(cur_batch.scheduled_decode)
        #    c. launch compute(cur_batch, fake_input_ids) on GPU stream
        #    d. result_queue.append((cur_batch, future_indices))
        #    e. if result_queue: process_batch_result(result_queue.popleft())  # post N-1
        #    f. launch sample on GPU stream  → writes into future_map
        ...
```

## 不变量

1. `OverlapScheduler.schedule()` 的输出与 normal Scheduler 同构，但额外携带 `future_indices`。
2. CPU 的 pre/post schedule 在概念上与 GPU 的 compute/sample 重叠（你 CPU 上模拟时可以用
   `concurrent.futures.ThreadPoolExecutor` 表达，重点验证依赖链不被破坏）。
3. 给定相同的 prompt 序列，OverlapScheduler 与 normal Scheduler 必须产出相同的 token 序列。

## 怎么验证

自己加一个 `patch/tests/test_overlap.py`（可选，不会被 patch-test 默认跑）：

```bash
cd labs/l20_vllm_scheduler_kv/patch
IMPL=starter pytest tests/test_overlap.py -v
```

## 写完之后你能做什么

- 解释 SGLang `event_loop_overlap` 与 `event_loop_normal` 的差异。
- 区分 multi-step scheduling（牺牲灵活性换吞吐）与 overlap scheduling（保留灵活性的 amortize）。
- 在 SGLang scheduler 调试报告里给出"如果切回 normal mode，吞吐跌多少"的估算。

## 配套阅读

- `github_repo/Awesome-ML-SYS-Tutorial/sglang/scheduler/readme.md` —— Scheduler 全文
- `github_repo/Awesome-ML-SYS-Tutorial/sglang/zero-overhead-scheduler/zero-overhead-batch-scheduler.md`
- 怜悯的 [SGLang Scheduler 技术变迁](https://zhuanlan.zhihu.com/p/1969077475129688722)
