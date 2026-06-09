# L21 Patch · vLLM-shaped Scheduler / KV Cache

## 你要交付什么

实现 `mini_infra/vllm/v1/core/sched/scheduler.py` 与 `kv_cache_manager.py` 的最小同构切片：

```python
class KVCacheManager:
    def allocate_slots(self, request_id, num_blocks) -> list[int]: ...
    def free(self, request_id) -> None: ...
    def snapshot(self) -> dict: ...

class Scheduler:
    def add_request(self, request) -> None: ...
    def schedule(self) -> SchedulerOutput: ...
```

## 不变量

1. `allocate_slots` 必须标记 KV block owner，KV 不够时抛错。
2. `free(request_id)` 必须释放该请求占用的所有 blocks。
3. Scheduler 把 waiting 请求 admit 到 running，受 `max_num_running_reqs` 和 KV blocks 限制。
4. running 请求未完成时进入 decode。
5. running 请求完成后进入 finished，并释放 KV blocks。

## 怎么验证

```bash
make patch-test M=l20_vllm_scheduler_kv
```

## 写完之后你能做什么

- 解释 vLLM scheduler 如何在 waiting/running/finished 之间移动请求。
- 解释 KV block pressure 如何影响 request admission。
- 给 AI tutor 画出 OpenAI entrypoint -> engine -> scheduler -> KV manager 的状态流。
