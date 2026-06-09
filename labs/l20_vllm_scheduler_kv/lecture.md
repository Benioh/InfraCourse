# 第 21 讲：vLLM Scheduler 与 KV Block Manager

这一讲讲推理服务的控制面。

一个请求已经到达 OpenAI-compatible API，并不代表模型马上开始生成 token。请求会先进入 engine，再进入 scheduler。scheduler 每一步都要做取舍：哪些请求可以进入 prefill，哪些请求继续 decode，哪些请求已经完成，哪些请求因为 KV cache 不够只能排队。KV Cache Manager 负责记录显存块的归属。两者配合起来，决定了服务的 TTFT、ITL、吞吐和显存压力。

这一讲的实验代码很小，只实现一个教学版 scheduler 和一个整数 KV block manager。代码小，是为了把系统合同讲清楚。真实 vLLM 会加入 token budget、chunked prefill、prefix cache、spec decode、KV connector、encoder cache 和 preemption；这些复杂功能最后仍然落在几个基本问题上：请求有没有资源运行，本 step 处理多少 token，已经完成的请求什么时候释放资源。

## 1. 这节课学完要能回答什么

学完这一讲，你应该能自然回答下面这些问题。

1. 一个 OpenAI chat completion 请求进入 vLLM 后，会经过哪些层？
2. prefill 和 decode 分别消耗什么资源，为什么长 prompt 会推高 TTFT？
3. KV cache 存的是什么，为什么它会限制并发？
4. scheduler 的 waiting、running、finished 三个状态分别代表什么？
5. KV block 不够时，请求应该留在队列里，还是直接失败？
6. scheduler output 要告诉 worker 什么信息？
7. 真实 vLLM 的 token budget、prefix cache、chunked prefill 和 preemption 分别在解决什么问题？
8. 当 TTFT 升高、waiting 队列增长、KV usage 接近满载时，应该按什么顺序排查？

这一讲的定位很明确：它属于 serving control plane。我们关心的是请求生命周期和资源调度，不深入 CUDA kernel、采样数学、HTTP 框架、模型权重加载和 tokenizer 细节。

## 2. 从一个线上现象讲起

假设你负责一个 vLLM 服务。监控里出现了这样的现象：

- HTTP 请求都返回 200，没有明显报错。
- GPU 利用率看起来不低。
- requests/sec 没有完全崩掉。
- 用户反馈首 token 等得很久。
- dashboard 上 waiting queue 越来越长。
- KV cache usage 接近上限。

这个时候只看 HTTP 层没有用。HTTP 200 只能说明入口接住了请求。只看 GPU 利用率也不够，GPU 忙不代表每个请求都公平地拿到了 first token。真正要看的，是请求进入 engine 之后，在 scheduler 和 KV manager 之间发生了什么。

推理服务的很多性能问题都长成这个样子：入口层很正常，模型也在跑，但调度层正在排队、等待 KV block、抢占旧请求，或者把长 prompt 切成多段 prefill。理解 scheduler，才能把“服务慢”拆成可定位的状态变化。

## 3. 一条请求的路径

先把请求路径放在脑子里。

```text
Client
  |
  v
OpenAI-compatible API
  |
  v
LLMEngine
  |
  v
Scheduler  <---->  KVCacheManager
  |
  v
Worker / Model Executor
  |
  v
Sampler / Output Processor
  |
  v
Response Stream
```

每一层负责的事情不一样。

| 层 | 主要职责 | 这一讲怎么用它 |
|---|---|---|
| OpenAI API | 接收 chat/completion 协议，解析 messages、model、sampling 参数 | 只作为入口，不把它当调度核心 |
| LLMEngine | 保存请求状态，驱动 step 循环，收集输出 | 看 `add_request` 和 `step` 怎么把请求交给 core |
| Scheduler | 决定本 step 处理哪些请求、多少 token、哪些请求结束或被抢占 | 本讲主角 |
| KVCacheManager | 分配、复用、释放 KV blocks，统计 usage | 本讲主角 |
| Worker / Executor | 根据 scheduler output 执行模型 forward | 这一讲只关心它收到什么 |
| Output Processor | 把内部输出转成用户可见的流式结果 | 用来理解 TTFT 和 finished output |

把这条路径记住，后面读源码就不会迷路。HTTP 层解决协议问题，engine 解决驱动问题，scheduler 解决资源和状态问题，KV manager 解决历史上下文的显存归属问题。

## 4. Prefill、Decode、TTFT 和 ITL

推理请求通常经历两个阶段。

**Prefill** 阶段处理完整 prompt。模型一次性读入 prompt token，计算 hidden states，并为每一层 attention 生成对应的 key/value。这个阶段并行度高，但长 prompt 会带来大量计算和 KV 写入。

**Decode** 阶段每次生成一个新 token。模型只接收上一步的新 token，但要读取历史上下文对应的 KV cache，才能让新 token attend 到过去所有 token。decode 的单步计算量小，访存和调度开销很关键。并发请求多时，decode 阶段尤其依赖 scheduler 把请求组织成高效 batch。

两个常见指标也在这里出现。

| 指标 | 含义 | 常见影响因素 |
|---|---|---|
| TTFT | Time To First Token，从请求到达服务到第一个 token 返回 | 排队、tokenization、prefill、KV block admission、prefix cache 命中 |
| ITL | Inter-Token Latency，相邻输出 token 的时间间隔 | decode batch 组织、KV 读取、模型执行、采样、调度间隔 |

长 prompt 更容易影响 TTFT，因为首 token 前要先完成 prompt 的计算和 KV 写入。长输出更容易影响端到端完成时间，因为 decode 要一轮一轮推进。KV pressure 会同时影响两者：waiting 请求进不来，running 请求也可能因为后续 block 不够被抢占或延迟。

## 5. KV Cache 到底是什么

Transformer 自回归生成时，每生成一个 token 都需要看历史上下文。没有 KV cache 时，模型每步都要重新计算历史 token 的 attention key/value，代价很高。有 KV cache 后，历史 token 的 key/value 张量被保留下来，decode 每步只计算新 token，并读取历史 KV。

KV cache 的收益很直接：减少 decode 阶段的重复计算。

KV cache 的代价也很直接：占显存。

粗略地看，KV cache 显存随这些量增长：

```text
batch / 并发请求数
  * sequence length / 上下文长度
  * num_layers / 层数
  * kv_heads / KV 头数
  * head_dim / 每个头的维度
  * dtype bytes / bf16、fp16、fp8 等精度
```

这解释了一个很常见的现象：模型参数装得下，不代表服务并发上得去。参数显存是静态的，KV cache 是随请求动态增长的。长上下文、高并发、大 `max_model_len` 会把 KV cache 推到主要瓶颈的位置。

## 6. 为什么要把 KV Cache 切成 block

真实服务里请求长度不同，生成长度也不同。直接给每个请求分一整段连续大显存，会浪费很多空间，也难以复用。vLLM 的思想是把 KV cache 管成 block。请求需要多少上下文，就占多少 block；请求结束后，把 block 还回池子。

你可以把 KV block 当作显存车位。

```text
Block 0: free
Block 1: request A
Block 2: request A
Block 3: request B
Block 4: free
Block 5: request C
```

KV Cache Manager 至少要能回答三件事：

1. 现在还有多少空闲 block？
2. 某个请求占了哪些 block？
3. 请求结束后，如何把这些 block 释放干净？

本关的教学版把真实 GPU block 简化成整数 id：

```python
@dataclass
class KVBlock:
    block_id: int
    owner_request_id: str | None = None
```

这个简化保留了最重要的资源合同：block 有归属，请求运行前要分配，结束后要释放。

## 7. Scheduler 的三个状态

教学版 scheduler 用三个状态就够讲清楚主线。

```text
waiting   ->   running   ->   finished
排队中         已占资源          已完成
```

**waiting** 表示请求已经进入系统，但还没有拿到运行资源。它可能在等 running slot，也可能在等 KV block。

**running** 表示请求已经被 admit，拿到了 KV block，可以参与本 step 的 prefill 或 decode。

**finished** 表示请求已经生成到 `max_tokens`，输出可以交回上层，KV block 应该释放。

这三个状态看起来简单，但足以覆盖许多服务问题：

| 现象 | 优先检查 |
|---|---|
| waiting 持续增长 | admission 是否被 running 上限或 KV block 卡住 |
| running 长时间不减少 | finished 判断、decode 推进、释放逻辑 |
| KV usage 只涨不降 | finished 后是否调用 `free` |
| TTFT 尾部变差 | waiting 时间、长 prompt、KV shortage、preemption |

## 8. Admission Control：请求什么时候能进 running

admission control 是 scheduler 的放行逻辑。教学版有两道门。

第一道门是 running 数量：

```python
len(self.running) < self.max_num_running_reqs
```

这限制活跃请求数，避免 scheduler 一口气把太多请求放进 running。

第二道门是 KV block：

```python
blocks_needed = max(1, (len(request.prompt_token_ids) + 15) // 16)
request.kv_blocks = self.kv_cache_manager.allocate_slots(
    request.request_id, blocks_needed
)
```

教学版约定每 16 个 prompt token 需要 1 个 KV block，至少 1 个。真实 vLLM 的计算更复杂，会考虑 block size、prefix cache hit、sliding window、chunked prefill、lookahead tokens、encoder tokens 和 KV connector，但判断逻辑的骨架一致：请求要进入运行态，必须有足够的 KV 容量承接它的上下文。

KV block 不够时，教学版捕获 `RuntimeError`，停止继续 admit，当前请求留在 waiting 队列里。

```python
try:
    request.kv_blocks = self.kv_cache_manager.allocate_slots(...)
except RuntimeError:
    break
```

这里的 `break` 很重要。它表示资源不够时，scheduler 保留请求，等待后续 step 有请求完成并释放 block。直接 `popleft` 会丢请求；直接报错退出会把服务层的资源压力变成用户可见失败。

## 9. schedule 一步到底做什么

教学版 `schedule()` 可以分成三段。

第一段，初始化本 step 的输出：

```python
scheduled_prefill = []
scheduled_decode = []
finished = []
```

第二段，尝试从 waiting admit 新请求：

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

第三段，处理 running 请求：

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

注意这里用 `list(self.running.items())`。循环中会删除 running 里的请求，所以先复制一份当前 items，避免迭代字典时修改字典。

最后返回：

```python
SchedulerOutput(scheduled_prefill, scheduled_decode, finished)
```

这份输出告诉 engine，本 step 哪些请求刚进入 prefill，哪些请求需要 decode，哪些请求已经完成。真实 vLLM 的输出字段更多，但目的相同：把 scheduler 的决策传给 worker 和 output processor。

## 10. Engine 如何使用 SchedulerOutput

MiniInfra 的 `LLMEngine.step()` 展示了最小驱动方式：

```python
scheduled = self.scheduler.schedule()
for request_id in scheduled.scheduled_decode:
    request = self.scheduler.running.get(request_id)
    if request is None:
        continue
    request.output_tokens.append(self._next_token(request))
```

engine 不自己决定谁运行。它调用 scheduler，拿到 `scheduled_decode`，然后给这些 running 请求追加一个 token。完成请求会通过 `scheduled.finished` 转成 `RequestOutput`。

这个结构很重要。很多同学第一次读推理框架，会把 engine、scheduler、worker 混在一起。一个清晰的分工是：

- engine 负责循环；
- scheduler 负责每一步选择谁；
- KV manager 负责资源；
- worker 负责模型计算；
- output processor 负责把内部状态转成用户输出。

## 11. 真实 vLLM 为什么没有简单的 prefill/decode 二分

教学版把请求分成 prefill 和 decode，方便理解。真实 vLLM 的 scheduler 更通用。源码注释里说，scheduler 关注的是 `num_computed_tokens` 追赶 `num_tokens_with_spec`。

可以这样理解：

```text
num_tokens_with_spec
  = prompt tokens
  + output tokens
  + speculative draft tokens

num_computed_tokens
  = 当前已经完成模型计算的 token 数
```

每个 step，scheduler 尝试给请求分配一部分新 token，让 `num_computed_tokens` 向 `num_tokens_with_spec` 靠近。这个模型能同时覆盖：

- 普通 prefill：prompt 还没算完；
- chunked prefill：长 prompt 分多步算；
- 普通 decode：每步补一个新 output token；
- speculative decoding：一次带上 draft tokens；
- prefix cache：有些 token 的 KV 已经命中；
- jump decoding 这类未来优化。

所以真实 vLLM 的抽象更像“token 进度追赶”，教学版抽象成“prefill/decode 状态”。两者的层级不同，目标一致：在资源预算内推进请求。

## 12. Token Budget 和 KV Block Budget

读真实 scheduler 时要分清两本账。

**Token budget** 限制本 step 最多调度多少 token。它主要影响一次模型 forward 的计算量和 batch 形状。

**KV block budget** 限制上下文历史能占多少显存。它主要影响请求能否被 admit，running 请求能否继续扩展上下文。

两者经常一起出现，实际限制的是两类资源。

一个请求可能有 token budget，但 KV block 不够，最后不能 schedule。也可能有 KV block，但本 step token budget 用完，只能等下一轮。生产调参时，`max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、`gpu_memory_utilization` 等参数会一起影响这两本账。

## 13. Prefix Cache、Chunked Prefill 和 Preemption

真实 vLLM 在教学版主线之外，还加了很多工程机制。先建立直觉，读源码时就不会被细节冲散。

**Prefix cache** 复用已有前缀的 KV blocks。多个请求共享相同 system prompt 或长上下文前缀时，scheduler 可以少算一部分 token，也少分一部分新 block。它改善 TTFT 和吞吐，但需要维护 block hash、引用计数和命中统计。

**Chunked prefill** 把长 prompt 的 prefill 切成多步。这样长 prompt 不会一次占满整个 step 的 token budget，短请求有机会穿插进来。它改善高并发下的调度公平性，但 scheduler 要管理更细的 token 进度。

**Preemption** 在 KV 压力过高时把某些 running 请求临时移回 waiting。真实 vLLM 会释放它的 KV 和 encoder cache，重置 computed tokens，并记录 preemption 次数。preemption 可以保护系统继续前进，但会带来重算和尾延迟抖动。

这三个机制都服务于同一个目标：在有限显存和有限 step budget 下，让请求生命周期可控。

## 14. SchedulerOutput 在真实 vLLM 里长什么样

教学版输出只有三个列表：

```python
SchedulerOutput(
    scheduled_prefill=[...],
    scheduled_decode=[...],
    finished=[...],
)
```

真实 vLLM 的 `SchedulerOutput` 包含更多字段：

| 字段 | 作用 |
|---|---|
| `scheduled_new_reqs` | 第一次被调度的新请求，worker 需要缓存它的请求数据 |
| `scheduled_cached_reqs` | worker 已经知道的请求，只发送本 step 的增量 |
| `num_scheduled_tokens` | 每个请求本 step 要处理多少 token |
| `total_num_scheduled_tokens` | 本 step 总 token 数 |
| `scheduled_spec_decode_tokens` | speculative decoding 的 draft token |
| `scheduled_encoder_inputs` | 多模态或 encoder-decoder 模型的 encoder 输入 |
| `finished_req_ids` | 本 step 结束的请求 |
| `preempted_req_ids` | 本 step 被抢占的请求 |
| `new_block_ids_to_zero` | 新分配 block 需要清零，避免旧数据污染 |

这些字段看起来多，背后仍然是同一件事：scheduler 把“本 step 的运行计划”交给执行层。执行层按计划跑模型，输出层再把结果变成用户可见的响应。

## 15. MiniInfra 为什么只考这一个切片

本关 patch 只实现这些函数：

```python
class KVCacheManager:
    def allocate_slots(self, request_id, num_blocks) -> list[int]: ...
    def free(self, request_id) -> None: ...
    def snapshot(self) -> dict: ...

class Scheduler:
    def add_request(self, request) -> None: ...
    def schedule(self) -> SchedulerOutput: ...
```

它考的是 serving scheduler 的最低合同，暂时收起 vLLM 的大部分生产细节。

| patch 行为 | 真实系统里的含义 |
|---|---|
| `allocate_slots` 标记 owner | 请求进入运行态前要占 KV |
| KV 不够抛错 | 资源不足要显式暴露 |
| `free` 释放 owner | 请求完成后归还 KV |
| `snapshot` 返回 usage 和 used blocks | 调试资源泄漏和错误共享 |
| waiting 到 running | admission control |
| running 到 decode | 本 step 参与生成 |
| finished 释放 KV | 请求生命周期闭环 |

所以 lab 只是一次出口检查。你真正要掌握的是：请求状态、KV 归属和调度输出之间的因果关系。

## 16. Drill 怎么读

`scripts/run_serving_drill.py` 会构造一个合成 workload，按 step 把请求加入 scheduler，然后反复调用 `schedule()`。

它记录几个关键指标：

| 指标 | 在 drill 中的定义 | 怎么解读 |
|---|---|---|
| `waiting` | 每步 waiting 队列长度 | admission 被卡住时会增长 |
| `kv_usage` | KV blocks 使用比例 | 接近 1.0 时说明 KV 压力很高 |
| `scheduled_prefill` | 本 step 新进入 running 的请求 | 观察首轮放行 |
| `scheduled_decode_count` | 本 step decode 请求数量 | 观察 decode 是否持续推进 |
| `finished` | 本 step 完成的请求 | 观察释放闭环 |
| `TTFT` | arrival step 到 first decode step 的差 | 观察首 token 等待 |
| `ITL` | first decode 到 finish 的平均 step/token | 观察 decode 节奏 |

如果 `kv_usage` 很高，`waiting` 也增长，系统正在受到 KV admission 限制。如果 `running` 一直有请求但 `finished` 很少，要看 decode 是否在追加 token，以及 `max_tokens` 判断是否正确。如果 `finished` 有增长但 `kv_usage` 不下降，要看 `free` 是否释放了 block owner。

## 17. 生产排查顺序

真实服务里遇到 TTFT 或吞吐问题，可以按这个顺序拆。

1. 确认 workload：prompt 长度、输出长度、并发、是否流式、是否有共享前缀。
2. 看队列：waiting 是否增长，arrival rate 是否超过服务能力。
3. 看 KV：usage、free blocks、preemption 次数、prefix cache hit。
4. 看 token budget：`max_num_batched_tokens` 是否限制了本 step 可处理 token。
5. 看 running：活跃请求是否长期不完成，decode 是否持续推进。
6. 看释放：finished 请求是否释放 KV 和 encoder cache。
7. 看参数：`max_model_len`、`max_num_seqs`、`gpu_memory_utilization`、chunked prefill、prefix caching。
8. 最后再比较框架差异：vLLM 和 SGLang 的接口不同，但 queue、prefill、decode、KV pressure 这些阶段仍然通用。

不要先猜模型慢。先把请求生命周期拆开，定位卡在哪张表、哪本预算、哪条状态迁移。

## 18. 本讲小结

这一讲把 vLLM serving 的控制面拆成了四个概念。

第一，请求到达 HTTP server 后，还要经过 engine，被 scheduler admit，再由 worker 执行。

第二，KV cache 是 decode 高效的基础，也是 serving 并发的硬约束。block manager 负责记录每个 block 的归属、释放和使用率。

第三，scheduler 的核心工作是用有限 running slot、token budget 和 KV block 推进请求生命周期。教学版用 waiting、running、finished 讲清主线，真实 vLLM 用 token 进度追赶模型覆盖更多优化。

第四，TTFT、ITL、KV usage、waiting queue 需要放在同一张图里看。只看吞吐或 HTTP 状态，很难判断请求到底卡在哪里。

接下来做 patch 时，记住这句话：每一行代码都应该维护请求生命周期和 KV 资源归属的一致性。测试只是检查这个合同有没有破。
