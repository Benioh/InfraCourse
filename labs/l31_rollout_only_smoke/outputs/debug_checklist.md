# Debug Checklist：L35 Async Rollout Pool

## 1. 固定现场

- 记录命令、配置、git commit、Python 环境、endpoint、request_count、max_new_tokens 和 max_concurrency。
- 保存 `command.sh`、`config.resolved.yaml`、`artifacts/rollouts.jsonl`、`metrics.jsonl`、`rl.log` 和 `report.md`。
- 标记运行类型：patch-test、CPU smoke、mock endpoint、真实 vLLM/SGLang endpoint 或 SLiME rollout。

## 2. 先查样本对齐

- rollouts 数量是否等于 prompt 数量。
- 每条样本是否保留 id、prompt、response、latency 和 reward 所需字段。
- 输出列表是否按输入 prompt 顺序排列。
- 异常是否向上传递，还是被写成空 response。

## 3. 再拆 rollout 慢

| 阶段 | 要看什么 | 常见动作 |
|---|---|---|
| 客户端 | `max_concurrency`、连接池、timeout、异常率 | 调并发上限或 HTTP client limit |
| 服务端队列 | TTFT、queue depth、running batch、KV cache | 降并发、缩短 max_new_tokens 或扩 rollout 资源 |
| 生成长度 | response_len、stop token、截断率 | 修 stop token、prompt 模板和输出上限 |
| 后处理 | reward parser、reward model、JSONL 写入 | 抽样本、跑 parser self-test、拆 reward 时间 |
| 同步 | weight sync 时间、policy version、generation 是否被打断 | 调 sync interval 或等待 generation 后同步 |

## 4. 对照源码

- `patch/reference/rollout_pool.py`：限流和保序的最小实现。
- `patch/tests/test_patch.py`：五个行为合同。
- `scripts/run_rollout_only.py`：本地 rollout schema、metrics 和 report。
- `github_repo/vllm/vllm/v1/engine/async_llm.py`：请求进入 AsyncLLM 并从 queue 返回输出。
- `github_repo/sglang/python/sglang/srt/managers/scheduler.py`：waiting queue、running batch 和 scheduler loop。
- `github_repo/slime/slime/rollout/sglang_rollout.py`：真实 rollout 的 semaphore、task、gather、reward 和排序。

## 5. 结束条件

- 可以用一个最小命令复现问题。
- 样本、metrics 和日志都能对应同一个 run。
- 能指出瓶颈位于客户端、服务端、生成长度、reward、weight sync 或训练侧。
- 结论写入 `rl_rollout_template.md`，并列出下一步只改一个变量的实验。
