# 源码带读：vLLM Scheduler 与 KV Block Manager

这份带读按“入口到调度”的顺序走。每一步只看少量代码，先抓主路径，再回头理解复杂分支。

建议先读 [lecture.md](lecture.md)，再打开源码。读源码时不要从文件顶部一路滚到底，直接按下面的锚点看。

## 0. 先记住本讲的源码地图

```text
mini_infra/vllm/entrypoints/openai/api_server.py
  -> mini_infra/vllm/v1/engine/llm_engine.py
    -> mini_infra/vllm/v1/core/sched/scheduler.py
      <-> mini_infra/vllm/v1/core/kv_cache_manager.py

github_repo/vllm/vllm/v1/engine/llm_engine.py
  -> github_repo/vllm/vllm/v1/core/sched/scheduler.py
    <-> github_repo/vllm/vllm/v1/core/kv_cache_manager.py
    -> github_repo/vllm/vllm/v1/core/sched/output.py
```

MiniInfra 负责把主线讲清楚。真实 vLLM 负责告诉你生产系统把这条主线扩展到什么程度。

## 1. MiniInfra OpenAI 入口：协议层只做入口整理

文件：[api_server.py](../../mini_infra/vllm/entrypoints/openai/api_server.py)

重点看第 15-23 行：

```python
prompt = "\n".join(message.get("content", "") for message in messages)
self.engine.add_request(request_id, prompt, max_tokens=max_tokens)
outputs = []
while self.engine.has_unfinished_requests():
    outputs = self.engine.step()
```

这里要得到三个结论。

第一，OpenAI-compatible facade 把 `messages` 拼成 prompt，然后把请求交给 engine。

第二，入口层没有决定请求什么时候 prefill，也没有决定哪个请求 decode。它只是把请求放进系统，并驱动 engine step 到完成。

第三，`mini_vllm_debug` 会把 `RequestOutput` 的内部信息带出来，方便教学时看 `kv_blocks` 和 `finished`。

读完这一段，先不要纠结 OpenAI 协议细节。它只是入口形状。

## 2. MiniInfra LLMEngine：engine 是循环驱动器

文件：[llm_engine.py](../../mini_infra/vllm/v1/engine/llm_engine.py)

重点看第 20-23 行：

```python
self.kv_cache_manager = KVCacheManager(num_blocks=num_kv_blocks)
self.scheduler = Scheduler(self.kv_cache_manager, max_num_running_reqs=max_num_running_reqs)
self.outputs: dict[str, RequestOutput] = {}
```

engine 持有 KV manager 和 scheduler。这个依赖关系很关键：scheduler 做决策时需要直接询问 KV manager。

再看第 25-28 行：

```python
def add_request(self, request_id: str, prompt: str, max_tokens: int = 4) -> None:
    self.scheduler.add_request(
        Request(request_id=request_id, prompt=prompt, max_tokens=max_tokens)
    )
```

engine 的 `add_request` 只是把请求包装成内部 `Request`，然后交给 scheduler。

最后看第 30-51 行：

```python
scheduled = self.scheduler.schedule()
for request_id in scheduled.scheduled_decode:
    request = self.scheduler.running.get(request_id)
    if request is None:
        continue
    request.output_tokens.append(self._next_token(request))
```

`step()` 先拿到 scheduler output，然后只给 `scheduled_decode` 里的 running 请求追加 token。完成请求通过 `scheduled.finished` 写成 `RequestOutput`。

读完这里要形成一个判断：engine 负责 step loop，scheduler 负责本 step 谁能动。

## 3. MiniInfra KVCacheManager：先读资源合同

文件：[kv_cache_manager.py](../../mini_infra/vllm/v1/core/kv_cache_manager.py)

重点看第 6-9 行：

```python
@dataclass
class KVBlock:
    block_id: int
    owner_request_id: str | None = None
```

一个 block 有编号和 owner。owner 为空表示可分配。

看第 24-27 行：

```python
@property
def usage(self) -> float:
    used = sum(block.owner_request_id is not None for block in self.blocks)
    return round(used / max(len(self.blocks), 1), 4)
```

usage 是已占 block 比例。这个指标在 drill 和真实服务里都很重要。

看第 29-38 行：

```python
free = [block for block in self.blocks if block.owner_request_id is None]
if len(free) < num_blocks:
    raise RuntimeError("KV cache exhausted")
allocated = free[:num_blocks]
for block in allocated:
    block.owner_request_id = request_id
return [block.block_id for block in allocated]
```

这是最小分配合同。空闲块够就标记 owner，空闲块不够就显式失败。

看第 43-46 行：

```python
for block in self.blocks:
    if block.owner_request_id == request_id:
        block.owner_request_id = None
```

释放时按 request id 清 owner。这个行为必须和 scheduler 的 finished 路径配合。

看第 53-67 行：

```python
return {
    "total_blocks": len(self.blocks),
    "usage": self.usage,
    "free_blocks": [...],
    "used_blocks": {...},
}
```

`snapshot()` 是调试入口。只看 usage 不够，`used_blocks` 能告诉你是否有重复占用、释放遗漏或 owner 错乱。

## 4. MiniInfra Scheduler：按三段读 schedule

文件：[scheduler.py](../../mini_infra/vllm/v1/core/sched/scheduler.py)

先看请求结构，第 9-17 行：

```python
@dataclass
class Request:
    request_id: str
    prompt: str
    max_tokens: int
    prompt_token_ids: list[str] = field(default_factory=list)
    output_tokens: list[str] = field(default_factory=list)
    kv_blocks: list[int] = field(default_factory=list)
```

这里有三个状态字段：

- `prompt_token_ids`：用 prompt 长度估算 KV block 需求；
- `output_tokens`：用输出长度判断是否 finished；
- `kv_blocks`：记录请求已经占到的 block。

再看 scheduler 状态，第 34-36 行：

```python
self.waiting: deque[Request] = deque()
self.running: dict[str, Request] = {}
self.finished: dict[str, Request] = {}
```

这三张表就是本关的主线。排查问题时，先看请求在哪张表。

看 `add_request`，第 38-40 行：

```python
request.prompt_token_ids = request.prompt.split()
self.waiting.append(request)
```

新请求进入 waiting。这里还没有分配 KV，因为排队请求不应该提前占显存。

看 `schedule` 的 admission， 第 46-57 行：

```python
while self.waiting and len(self.running) < self.max_num_running_reqs:
    request = self.waiting[0]
    blocks_needed = max(1, (len(request.prompt_token_ids) + 15) // 16)
    try:
        request.kv_blocks = self.kv_cache_manager.allocate_slots(
            request.request_id, blocks_needed
        )
    except RuntimeError:
        break
    self.waiting.popleft()
    self.running[request.request_id] = request
    scheduled_prefill.append(request.request_id)
```

这一段同时检查 running 上限和 KV block。分配成功才从 waiting 移到 running。分配失败就停下，保留请求。

最后看 running 处理，第 58-65 行：

```python
for request_id, request in list(self.running.items()):
    if len(request.output_tokens) >= request.max_tokens:
        self.finished[request_id] = request
        self.kv_cache_manager.free(request_id)
        del self.running[request_id]
        finished.append(request_id)
    else:
        scheduled_decode.append(request_id)
```

这段包含完整释放闭环：写入 finished，释放 KV，从 running 删除，输出 finished id。少一步都会出问题。

## 5. Patch Reference：看学生要补的最小合同

文件：[patch/reference/scheduler.py](patch/reference/scheduler.py)

参考解和 MiniInfra 主线保持同构。重点看：

- 第 26-35 行：`allocate_slots`
- 第 37-40 行：`free`
- 第 42-54 行：`snapshot`
- 第 84-86 行：`add_request`
- 第 88-115 行：`schedule`

读 reference 时不要背代码。用下面这个顺序复述：

1. 新请求先进 waiting。
2. `schedule()` 先尝试 admit waiting。
3. admission 先看 running 上限，再算 KV block。
4. KV 分配成功后请求进入 running。
5. running 请求如果还没到 `max_tokens`，进入 `scheduled_decode`。
6. 生成完成后进入 finished，释放 KV。

能流畅复述这一段，再写 patch 就不会变成猜测试。

## 6. 真实 vLLM LLMEngine：看 add_request 和 step

文件：[llm_engine.py](../../github_repo/vllm/vllm/v1/engine/llm_engine.py)

重点看第 209-268 行。

`add_request` 做了这些事：

1. 校验 request id。
2. 把 raw prompt 或 EngineInput 处理成 `EngineCoreRequest`。
3. 把请求交给 output processor 建立输出状态。
4. 调用 `self.engine_core.add_request(request)`。

这里的主线和 MiniInfra 一致：入口进入 engine，engine 把内部请求交给 core。

再看第 287-325 行。

真实 `step()` 的节奏是：

1. 从 `engine_core.get_output()` 拿内部输出。
2. `output_processor.process_outputs(...)` 把输出转成 RequestOutput。
3. abort 已经被 stop string 终止的请求。
4. 记录 scheduler stats。
5. 返回用户可见 output。

真实 vLLM 把 scheduler 放在 EngineCore 内部，外层 LLMEngine 看到的是 core output。读这一段时抓住分层关系即可。

## 7. 真实 vLLM Scheduler：先读 schedule 顶部注释

文件：[scheduler.py](../../github_repo/vllm/vllm/v1/core/sched/scheduler.py)

重点看第 352-385 行。

源码注释说明真实 scheduler 的核心模型：每个请求有 `num_computed_tokens` 和 `num_tokens_with_spec`，每一步 scheduler 尝试分配 token，让 computed 追赶 tokens with spec。

这句话把许多复杂功能串起来：

- prompt 没算完时，是 prefill；
- prompt 分段算时，是 chunked prefill；
- 已经开始生成时，是 decode；
- 带 draft token 时，是 spec decode；
- 命中 prefix cache 时，一部分 token 已经有 KV；
- 未来 jump decoding 也能放进这个框架。

所以读真实 scheduler 时，不要急着找一个叫 `prefill_phase` 的分支。真实实现更关注 token 进度和资源预算。

## 8. 真实 vLLM Scheduler：看 waiting admission

继续看 [scheduler.py](../../github_repo/vllm/vllm/v1/core/sched/scheduler.py) 第 744-769 行。

关键点有两个。

第一个是 `can_fit_full_sequence`：

```python
if (
    self.scheduler_reserve_full_isl
    and not self.kv_cache_manager.can_fit_full_sequence(...)
):
    ...
    break
```

这个 gate 用来避免 chunked prefill 只看当前小段，从而过度 admit 长请求。长 prompt 可能第一段放得下，完整上下文放不下。生产系统要更保守地看完整序列容量。

第二个是 `allocate_slots`：

```python
new_blocks = self.kv_cache_manager.allocate_slots(...)
if new_blocks is None:
    ...
    break
```

真实 vLLM 里分配失败返回 `None`，教学版里分配失败抛 `RuntimeError`。接口不同，资源语义一样：KV 不够时不能继续 admit 当前请求。

## 9. 真实 KVCacheManager：看 usage、fit 和 allocate

文件：[kv_cache_manager.py](../../github_repo/vllm/vllm/v1/core/kv_cache_manager.py)

先看第 106-169 行。

真实 `KVCacheManager` 初始化时接收 `kv_cache_config`、`max_model_len`、`hash_block_size`、`max_num_batched_tokens` 等配置，内部持有 coordinator 和 block pool。`usage` 直接来自 block pool：

```python
return self.block_pool.get_usage()
```

这和教学版 `used / total` 对齐。

再看第 225-263 行的 `can_fit_full_sequence`。

它会根据请求 token 数、已计算 token、prefix cache 命中、encoder token 等信息，计算完整序列还需要多少 block，再和 free blocks 比较。这个函数体现了生产 admission 的谨慎性。

最后看第 265-380 行的 `allocate_slots` 开头。

真实分配要处理这些情况：

- prefix cached tokens；
- connector 外部已有 KV；
- speculative lookahead tokens；
- encoder-decoder cross-attention；
- sliding window 释放；
- block 引用计数；
- 异步 KV transfer。

先不要陷进每个分支。抓住一条主线：这个函数在给请求的新 token 和已有上下文安排 KV slot。

## 10. 真实 preemption 和 finished 释放

文件：[scheduler.py](../../github_repo/vllm/vllm/v1/core/sched/scheduler.py)

看第 965-985 行 `_preempt_request`。

抢占请求时，vLLM 会：

1. 释放 KV cache；
2. 释放 encoder cache；
3. 把状态改成 `PREEMPTED`；
4. 重置 `num_computed_tokens`；
5. 清掉 spec token；
6. 增加 preemption 计数；
7. 放回 waiting。

抢占不是普通完成，也不是用户失败。它是 scheduler 在 KV 压力下让系统继续推进的一种策略。

再看第 1826-1847 行 `_free_request` 和 `_free_blocks`。

finished 请求会进入释放路径，最终调用：

```python
self.kv_cache_manager.free(request)
del self.requests[request.request_id]
```

这和教学版 finished 路径的核心一致：完成请求必须释放 KV，并从活跃请求表里移除。

## 11. 真实 SchedulerOutput：看调度决策如何交给 worker

文件：[output.py](../../github_repo/vllm/vllm/v1/core/sched/output.py)

重点看第 178-253 行。

真实 `SchedulerOutput` 字段较多，读的时候按用途分组。

请求数据：

- `scheduled_new_reqs`
- `scheduled_cached_reqs`

token 计划：

- `num_scheduled_tokens`
- `total_num_scheduled_tokens`
- `scheduled_spec_decode_tokens`

多模态 / encoder：

- `scheduled_encoder_inputs`
- `free_encoder_mm_hashes`

生命周期：

- `finished_req_ids`
- `preempted_req_ids`

KV 安全：

- `new_block_ids_to_zero`
- `kv_connector_metadata`
- `ec_connector_metadata`

这份结构就是 scheduler 给执行层的计划书。MiniInfra 的三个列表是它的极简版本。

## 12. 读完源码后的自检

读完这条路径后，合上文件，尝试回答：

1. OpenAI API 层把请求交给谁？
2. MiniInfra engine 的 `step()` 为什么要先调用 `scheduler.schedule()`？
3. 新请求为什么在 `add_request` 阶段只进 waiting？
4. `blocks_needed` 为什么由 prompt token 数推导？
5. KV 不够时，教学版为什么保留 waiting 请求？
6. finished 请求释放 KV 时，哪些状态要一起更新？
7. 真实 vLLM 为什么用 `num_computed_tokens` 追赶 `num_tokens_with_spec`？
8. `token_budget` 和 `KV block budget` 分别限制什么？
9. prefix cache、chunked prefill、preemption 分别解决哪类 serving 问题？
10. 如果 TTFT 变差，你会在哪些源码字段或指标里找证据？

这些问题能答顺，再进入 patch。

