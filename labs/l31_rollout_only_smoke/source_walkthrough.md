# 源码带读：L35 Async Rollout Pool

这份带读按“本地 smoke -> patch -> vLLM/SGLang -> SLiME”的顺序走。先读最小样本生产线，再看 async pool 的行为合同，最后把同样的限流、保序和请求调度概念放进真实框架。

## 0. 源码地图

```text
labs/l31_rollout_only_smoke/scripts/run_rollout_only.py
mini_infra/rl/rollout.py
mini_infra/rl/reward.py

labs/l31_rollout_only_smoke/patch/starter/rollout_pool.py
labs/l31_rollout_only_smoke/patch/reference/rollout_pool.py
labs/l31_rollout_only_smoke/patch/tests/test_patch.py

github_repo/vllm/vllm/v1/engine/async_llm.py
github_repo/vllm/vllm/v1/engine/output_processor.py
github_repo/sglang/python/sglang/srt/managers/scheduler.py

github_repo/slime/train_async.py
github_repo/slime/slime/rollout/sglang_rollout.py
github_repo/slime/slime/utils/http_utils.py
```

## 1. 本地 rollout-only smoke

文件：[scripts/run_rollout_only.py](scripts/run_rollout_only.py)

先看第 53-68 行。`mock_response` 构造 response、latency、token 估计和 `reward_input_ready`。它说明本地 smoke 的目标是 schema 和 artifact，不是模型质量。

再看第 86-99 行。这里准备 run 目录，写 command、prediction 和 resolved config。`actor_update=False`、`weight_sync=False` 是边界声明。

最后看第 110-122 行和第 130-142 行。前者写 rollout metrics，后者把实验矩阵和迁移边界写进 report。

文件：[mini_infra/rl/rollout.py](../../mini_infra/rl/rollout.py)

第 20-32 行展示 MiniInfra rollout 行结构：prompt、response、target、latency 和 reward。第 39-51 行展示 run 入口如何写 metrics。第 57-61 行写 artifacts 和 report。

文件：[mini_infra/rl/reward.py](../../mini_infra/rl/reward.py)

第 12-18 行提取最终数字。第 21-33 行用 prediction 和 target 计算 reward，并保留 prediction/target 的最终值。

## 2. Patch controller

文件：[patch/starter/rollout_pool.py](patch/starter/rollout_pool.py)

第 19-28 行是初始化 TODO：保存 `generate_fn` 和 `max_concurrency`。第 30-42 行是 rollout TODO：空输入、semaphore、bounded coroutine、gather。

文件：[patch/reference/rollout_pool.py](patch/reference/rollout_pool.py)

第 9-16 行保存状态。第 18-28 行展示完整实现：空列表直接返回，`async with sem` 限流，`asyncio.gather` 保序返回。

文件：[patch/tests/test_patch.py](patch/tests/test_patch.py)

按顺序读五组测试：

- 第 23-31 行：基本输出来自 `generate_fn`。
- 第 34-46 行：不同延迟下仍保持输入顺序。
- 第 49-66 行：计数器记录最大并发，不能超过上限。
- 第 69-77 行：空输入返回空列表。
- 第 80-90 行：`generate_fn` 抛错时继续向外抛。

## 3. vLLM 对照

文件：[github_repo/vllm/vllm/v1/engine/async_llm.py](../../github_repo/vllm/vllm/v1/engine/async_llm.py)

第 280-292 行是 `add_request` 的入口签名。第 367-379 行创建 output collector 并把 request 加入引擎。第 521-537 行是 `generate` 的 async generator 入口。第 553-565 行调用 `add_request`；第 568-580 行从 per-request queue 取输出并 yield；第 583-611 行展示取消、校验错误和输入流错误如何向上传播。

文件：[github_repo/vllm/vllm/v1/engine/output_processor.py](../../github_repo/vllm/vllm/v1/engine/output_processor.py)

第 643-651 行创建 `RequestOutput`。第 655-660 行把输出放入 request queue 或返回给同步 engine。它对应 L35 里“结果要回到自己的请求”的保序问题。

## 4. SGLang scheduler 对照

文件：[github_repo/sglang/python/sglang/srt/managers/scheduler.py](../../github_repo/sglang/python/sglang/srt/managers/scheduler.py)

第 993-999 行初始化 waiting queue、running batch 和当前 batch。第 1468-1486 行展示 normal event loop：收请求、处理输入、取 batch、运行 batch、处理结果。第 1508-1528 行展示 overlap loop 里的接收、取 batch 和运行 batch。

读完后要能说清：客户端 `max_concurrency` 只控制请求入口，服务端 scheduler 才决定运行 batch。

## 5. SLiME rollout 对照

文件：[github_repo/slime/train_async.py](../../github_repo/slime/train_async.py)

第 17-29 行创建 rollout manager、训练模型，并把 actor 权重推到 rollout。第 34-43 行提前启动下一轮 rollout。第 45-53 行训练 actor/critic。第 69-79 行在更新 rollout 权重前等待 generation，同步后继续 eval 和清理。

文件：[github_repo/slime/slime/rollout/sglang_rollout.py](../../github_repo/slime/slime/rollout/sglang_rollout.py)

第 94-96 行创建 semaphore。第 136-149 行提交 generation task。第 239-277 行在 semaphore 内执行 generation。第 323-333 行为同组样本创建 task 并 gather。第 420-452 行持续提交任务、等待先完成的 task 并过滤样本。第 460-470 行中止剩余请求、按样本 index 排序，并重置状态。

文件：[github_repo/slime/slime/utils/http_utils.py](../../github_repo/slime/slime/utils/http_utils.py)

第 201-213 行根据 rollout concurrency 创建 HTTP client 连接上限。第 239-266 行创建 distributed POST actor。第 275-293 行说明直接 await Ray ObjectRef，避免轮询造成尾延迟。

## 可以先跳过

- vLLM 的 LoRA、reasoning parser 和 streaming input 细节。
- SGLang overlap loop 的 strict memory check 和 grammar sampling 分支。
- SLiME 的 multimodal、dynamic filter、partial rollout 和 abort 收集细节。

这些分支会影响生产行为，但第一次阅读先把请求入口、限流、结果回传和样本排序读通。

## 自检问题

1. `RolloutPool` 的保序来自 `Semaphore` 还是 `gather`？
2. 客户端 `max_concurrency` 和服务端 batch size 有什么区别？
3. vLLM 的 request output 如何回到对应请求？
4. SGLang scheduler 的 waiting queue 和 running batch 分别代表什么状态？
5. SLiME 为什么在更新 rollout 权重前等待正在生成的任务结束？
