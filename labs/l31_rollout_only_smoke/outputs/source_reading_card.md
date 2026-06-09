# Source Reading Card：L35 Async Rollout Pool

## 主路径

1. `labs/l31_rollout_only_smoke/scripts/run_rollout_only.py`：本地 rollout schema、metrics 和报告边界。
2. `labs/l31_rollout_only_smoke/patch/starter/rollout_pool.py`：学生实现的并发限流入口。
3. `labs/l31_rollout_only_smoke/patch/reference/rollout_pool.py`：`Semaphore + gather` 的最小实现。
4. `labs/l31_rollout_only_smoke/patch/tests/test_patch.py`：基本输出、保序、限流、空输入和异常透传。
5. `github_repo/vllm/vllm/v1/engine/async_llm.py`：请求进入 AsyncLLM，输出从 queue 返回。
6. `github_repo/sglang/python/sglang/srt/managers/scheduler.py`：waiting queue、running batch 和 scheduler loop。
7. `github_repo/slime/train_async.py`：rollout manager、actor/critic train 和 weight sync。
8. `github_repo/slime/slime/rollout/sglang_rollout.py`：真实 rollout 的 semaphore、task、gather、reward 和排序。

## 每段要得到的结论

| 文件 | 读完后要能说明 |
|---|---|
| `run_rollout_only.py` | 本地 smoke 证明 schema 和 artifact，不证明真实服务性能 |
| `patch/reference/rollout_pool.py` | semaphore 负责限流，gather 负责按输入顺序返回 |
| `patch/tests/test_patch.py` | async 代码的边界需要用延迟、计数器和异常一起测 |
| `async_llm.py` | 请求被加入引擎后，输出通过 per-request queue 回到调用者 |
| `scheduler.py` | 服务端根据 waiting/running 状态组织真实 batch |
| `train_async.py` | rollout、训练、保存、weight sync 和 eval 交替推进 |
| `sglang_rollout.py` | 生产版 rollout 还要处理 reward、过滤、abort 和样本排序 |

## 自检

- `max_concurrency` 限制什么，不限制什么？
- 为什么 `generate_fn` 必须放在 `async with sem` 内部？
- 为什么完成顺序不能直接作为训练样本顺序？
- 接真实 endpoint 后，哪些指标可以判断 server queue 或生成长度成为瓶颈？
