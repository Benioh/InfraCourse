# L35 · Async Rollout Pool：并发上限、保序返回与 Rollout-only Smoke

<!-- LECTURE_FIRST_START -->

L35 进入 RL rollout 系统的请求侧。它承接 L31 的 KL/reward baseline、L33 的偏好损失和 L34 的 co-locate 资源边界；这一讲只看生成样本如何被安全地并发提交：客户端用 `asyncio.Semaphore` 限制在途请求数，用 `asyncio.gather` 保留输入顺序，再把 rollout response、latency 和 reward-ready 字段落盘。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L35 在 RLHF / rollout systems 主线中的位置。
2. 读 [lecture.md](lecture.md)：从 rollout 数据流、async 并发、服务端动态 batching 讲到 SLiME 对照和排障。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按本地 smoke、patch、vLLM/SGLang、SLiME 主路径阅读。
4. 做 quiz：确认并发上限、保序、reward-ready schema、weight sync 边界和 debug 顺序。
5. 做 patch：实现 `RolloutPool` 并通过行为测试。
6. 跑 smoke：生成本地 rollout artifacts 和 metrics。
7. 填写 [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md)，记录一次 rollout-only 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | RLHF and rollout systems |
| 核心风险 | 串行 rollout 浪费服务端调度能力，无上限并发压垮 server，完成顺序打乱训练样本 |
| 关键机制 | `asyncio.Semaphore`、`asyncio.gather`、保序返回、rollout schema、reward-ready 检查 |
| 源码落点 | patch controller、本地 rollout smoke、vLLM AsyncLLM、SGLang scheduler、SLiME rollout |
| lab 检验 | 基本输出、保序、并发上限、空输入、异常透传 |

## Patch 闭环

```bash
cat labs/l31_rollout_only_smoke/patch/task.md
$EDITOR labs/l31_rollout_only_smoke/patch/starter/rollout_pool.py
make patch-test M=l31_rollout_only_smoke
```

参考实现验收：

```bash
IMPL=reference make patch-test M=l31_rollout_only_smoke
```

Smoke：

```bash
python labs/l31_rollout_only_smoke/scripts/run_rollout_only.py --run-id l35_smoke
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 rollout 慢、顺序错位、server queue 积压和 reward-ready 缺失 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习本地 smoke、patch、vLLM/SGLang 和 SLiME 主路径 |
| [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) | 记录 rollout-only run 的配置、样本、指标、判断和下一步 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L35 后进入 L36 SLiME RL Core。下一讲会把 rollout 数据、actor update 和 weight sync 放进完整训练循环。
