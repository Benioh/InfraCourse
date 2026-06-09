# L35 · Async Rollout Pool

RLHF/PPO 训练需要不断生产新样本：拿一批 prompt，让当前 policy 生成 response，再把 response 交给 reward、logprob、KL 和 advantage 计算。这个阶段常被叫做 rollout。L35 只处理 rollout 的请求侧：如何并发提交 prompt，如何限制在途请求数，如何保证输出顺序仍和输入 prompt 对齐。

完整 RL 训练还包含 reward model、reference model、actor update、critic update 和 weight sync。L35 不做这些。我们先把一个小而常见的客户端合同写准：`RolloutPool(generate_fn, max_concurrency).rollout(prompts)`。

## 1. 本讲目标

- 解释 RL rollout 数据从 prompt 到 response、reward-ready artifact 的路径。
- 区分客户端 `max_concurrency`、服务端 dynamic batching 和训练 batch。
- 手写 `asyncio.Semaphore + asyncio.gather` 的并发上限实现。
- 用测试验证保序、限流、空输入和异常透传。
- 对照 vLLM、SGLang 和 SLiME，说明真实系统多出的 queue、scheduler、reward 和 weight sync。

## 2. Rollout 在 RLHF 里的位置

一次 rollout 的输出不是普通文本列表。训练侧通常需要这些字段：

- prompt：原始输入。
- response：policy 生成的文本或 token。
- latency / token count：判断 rollout 是否成为瓶颈。
- reward input：reward parser 或 reward model 需要的字段。
- logprob / ref logprob：后续 PPO/GRPO 计算 KL 和 importance ratio。
- sample id / index：用来保持数据对齐和复盘。

L35 的本地 smoke 只保存其中一部分：prompt、response、expected answer、latency、token 估计和 `reward_input_ready`。这已经足够验证 schema 和 artifact 习惯。真实训练会再补 old logprob、ref logprob、reward、advantage 和 policy version。

样本对齐是 rollout 的底线。如果 prompt A 的 response 被放到 prompt B 的位置，reward、KL 和 advantage 都会绑定到错误样本。这样的错误不一定立刻报错，训练曲线却会变得难以解释。

## 3. 为什么用 async rollout

vLLM 和 SGLang 这类服务端推理引擎已经有 scheduler。客户端把请求并发提交过去，服务端根据 waiting queue、running batch、KV cache、decode slot 和 stop 条件组织真正的 GPU batch。

客户端手动把 prompt 切成固定 batch，有两个问题。第一，长输出会拖住短输出，同一个 batch 要等尾部请求结束。第二，服务端已经有动态调度能力，客户端硬凑 batch 会减少它根据实时状态重组请求的空间。

串行调用也不合适。一个 prompt 完成后才发下一个，服务端很容易长期处在低并发状态，GPU forward 的 batch 变小，decode slot 空着。async rollout 的目标是让服务端保持可用请求，但又不让请求洪峰把 queue、连接池、KV cache 或显存打满。

因此客户端需要一个旋钮：`max_concurrency`。它限制在途请求数，不等于 GPU batch size。调它时要看 server queue、TTFT、decode latency、OOM、HTTP 错误、stop token 和 response length。

## 4. `Semaphore + gather` 的机制

本讲 patch 的参考结构如下：

```python
sem = asyncio.Semaphore(self.max_concurrency)

async def _bounded(prompt: str) -> str:
    async with sem:
        return await self.generate_fn(prompt)

tasks = [_bounded(prompt) for prompt in prompts]
return await asyncio.gather(*tasks)
```

`Semaphore` 管并发入口。`async with sem` 进入时申请一个许可，离开时释放许可。只要 `generate_fn` 放在这个上下文内部，同一时刻执行的生成协程就不会超过 `max_concurrency`。

`gather` 管结果顺序。即使 `"b"` 比 `"a"` 更早完成，`asyncio.gather(task_a, task_b, task_c)` 返回的列表仍按传入任务的顺序排列。这个合同正好满足 rollout 的样本对齐要求。

异常不要吞。某个 `generate_fn` 抛出 `ValueError`、HTTP error 或 timeout 时，默认 `gather` 会把异常传给调用者。训练 driver 可以在上层决定重试、降并发或中止 run。L35 的 patch 要求异常透传，是为了防止坏样本悄悄进入 buffer。

## 5. Patch 的边界

`RolloutPool` 的输入是一组字符串 prompt。内部状态只有 `generate_fn`、`max_concurrency` 和每次调用时创建的 semaphore。输出是字符串列表。

它不负责：

- HTTP client 的连接池。
- vLLM/SGLang server 的调度策略。
- reward model forward。
- actor/ref logprob。
- weight sync。
- 失败重试策略。

这些边界并不是偷懒。它们让单元测试可以专注检查四件事：能并发、会限流、能保序、异常不被吞。

## 6. 本地 Rollout-only Smoke

`scripts/run_rollout_only.py` 用四条 toy 数学题模拟 rollout。`mock_response` 会生成带 `Final answer:` 的 response，并记录 `ttft_ms`、`latency_ms`、`output_tokens_est` 和 `reward_input_ready`。

脚本会写出：

- `artifacts/rollouts.jsonl`：每条 rollout 样本。
- `metrics.jsonl`：`rollouts_per_sec`、`avg_ttft_ms`、`avg_output_tokens`、`mock_server`、`actor_update`、`weight_sync`。
- `rl.log`：简短运行日志。
- `report.md`：本地 smoke 的目标、指标和迁移边界。

报告会明确 `actor_update=False` 和 `weight_sync=False`。这句话很重要：本地 smoke 证明 schema、artifact 和指标字段可用，不证明真实 SGLang 性能，也不证明训练循环已经稳定。

## 7. vLLM 和 SGLang 对照

vLLM 的 `AsyncLLM.generate` 会先调用 `add_request`，创建 request 的输出 collector，然后从 per-request queue 里取出 `RequestOutput` 并 yield 给调用者。后台 output handler 从 engine core 拉结果，再推回对应 queue。

SGLang scheduler 的核心状态包括 `waiting_queue`、`running_batch`、`cur_batch` 和 `last_batch`。normal event loop 会接收请求、处理输入、取下一批 batch、运行 batch、处理结果。overlap loop 还会把 CPU 处理和 GPU 计算做一定重叠。

这说明客户端的 async pool 只是入口层。真正的 continuous batching、KV cache 资源管理、prefill/decode 切换和 stop 条件处理发生在服务端。客户端如果没有限流，服务端 queue 会积压；客户端如果没有并发，服务端 scheduler 拿不到足够请求。

## 8. SLiME 对照

SLiME 的 async train loop 会创建 rollout manager、actor、critic，先把 actor 权重推给 rollout，然后提前启动下一轮 rollout，在当前 rollout 数据上训练 actor/critic。到 `update_weights_interval` 时，它会等待正在生成的 rollout 结束，再更新 rollout 侧权重，避免 generation 中途切换 policy。

SLiME 的 SGLang rollout 代码里也有同构组件。`GenerateState` 根据 server concurrency 和 rollout GPU 数创建 semaphore；`generate_and_rm` 在 semaphore 内调用 generation；`generate_and_rm_group` 用 task 和 gather 处理同组样本；`generate_rollout_async` 持续提交任务、等待先完成的 task、过滤样本，最后按样本 index 排序。

L35 的小 patch 是这个流程的单机最小版。真实代码多了 Ray actor、HTTP client、reward model、dynamic filter、abort、global dataset 和 metrics，但限流、保序、样本身份这三个概念没有变。

## 9. 训练和推理解耦

RL 训练通常把 train engine 和 rollout engine 分开看。训练侧需要参数、梯度、优化器状态、activation 和通信；推理侧主要需要参数、KV cache、scheduler 和请求队列。两边的资源压力不同，因此框架会选择 co-locate 或 disaggregate。

Co-locate 让训练和 rollout 共享同一组 GPU，资源紧张时更省卡，但要处理 pause/resume、offload/upload 和显存碎片。Disaggregate 让训练和 rollout 常驻不同资源组，吞吐调度更清楚，但资源利用率和 weight sync 协议更难。

L35 主线不实现 placement。它先把 rollout 请求的客户端合同写准。后续进入完整 SLiME 主线时，再把 `max_concurrency`、rollout GPU 数、server concurrency 和 weight sync interval 放在一起调。

## 10. Debug 路线

rollout 慢时，先分阶段看：

1. 客户端是否按预期并发提交请求。
2. server queue 是否积压，TTFT 是否变大。
3. `max_new_tokens`、stop token 和 response length 是否合理。
4. HTTP client 连接数是否成为瓶颈。
5. reward model 或 parser 是否拖住后处理。
6. weight sync 是否挡住 generation。

reward collapse 时，先抽样本。看 response 是否有最终答案，`reward_input_ready` 是否为 true，parser 是否能提取数字，再看 KL、entropy、policy version 和 weight sync 日志。

接真实 endpoint 后，一次实验只改一个变量，例如只改 `max_concurrency`，或者只改 `max_new_tokens`。失败 run 要保留 command、config、rollouts、metrics 和日志。吞掉异常会让训练继续写入坏数据，这通常比直接失败更难处理。

## 11. Lab 验收

patch 命令：

```bash
IMPL=reference make patch-test M=l31_rollout_only_smoke
```

smoke 命令：

```bash
python labs/l31_rollout_only_smoke/scripts/run_rollout_only.py --run-id l35_smoke
```

完成 L35 后，学生应该能从一个 toy async pool 讲到真实 RL rollout：客户端限流，服务端动态 batching，样本保序，artifact 可查，异常向上暴露。
