# 源码阅读卡：vLLM Scheduler 与 KV Cache

这张卡用于快速回忆本讲的源码主路径。读源码时按顺序走，先抓主路径，再看分支。

## 1. MiniInfra 主线

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `mini_infra/vllm/entrypoints/openai/api_server.py` | `create_chat_completion` 如何拼 prompt、调用 engine | OpenAI 层只是入口和协议形状 |
| `mini_infra/vllm/v1/engine/llm_engine.py` | `add_request`、`step`、`RequestOutput` | engine 驱动循环，调度决策来自 scheduler |
| `mini_infra/vllm/v1/core/sched/scheduler.py` | `waiting`、`running`、`finished`、`schedule` | 请求生命周期由三张表维护 |
| `mini_infra/vllm/v1/core/kv_cache_manager.py` | `allocate_slots`、`free`、`snapshot` | KV block 有 owner，完成后必须释放 |

## 2. 真实 vLLM 主线

| 文件 | 只看什么 | 得到什么结论 |
|---|---|---|
| `github_repo/vllm/vllm/v1/engine/llm_engine.py` | `add_request`、`step` | 外层 engine 处理输入和输出，调度在 core 内部 |
| `github_repo/vllm/vllm/v1/core/sched/scheduler.py` | `schedule` 顶部注释、running/waiting 调度、preemption、finished free | 真实 scheduler 用 token 进度追赶模型统一 prefill、decode、chunked prefill 和 spec decode |
| `github_repo/vllm/vllm/v1/core/kv_cache_manager.py` | `usage`、`can_fit_full_sequence`、`allocate_slots` | 真实 KV 分配要考虑 prefix cache、sliding window、connector、lookahead 和 encoder tokens |
| `github_repo/vllm/vllm/v1/core/sched/output.py` | `SchedulerOutput` 字段 | scheduler output 是交给 worker 的本 step 计划 |

## 3. 读源码时的提问顺序

1. 这个函数在请求生命周期的哪一层？
2. 它读写的是 waiting、running、finished，还是 KV blocks？
3. 它改变了 token 进度、KV 归属，还是输出状态？
4. 失败路径是什么，资源会不会释放？
5. 这个分支影响 TTFT、ITL、throughput，还是显存压力？

## 4. 本讲最小不变量

- 新请求先进入 waiting。
- 请求进入 running 前必须拿到 KV blocks。
- KV block 不够时，请求留在 waiting。
- running 请求每 step 才进入 decode。
- 请求完成后进入 finished。
- finished 请求必须释放 KV blocks。
- scheduler output 要能告诉执行层本 step 处理谁。

