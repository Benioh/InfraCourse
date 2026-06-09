# L40 · GAE Chunked Parallel：Boundary State and Long-Context RL

<!-- LECTURE_FIRST_START -->

本讲处理 PPO/GRPO 训练里的 GAE 递推。L39 讲 critic-free advantage；L40 看带 value critic 的路线：如何从 reward/value 递推出 token-level advantage，并把长上下文下的时间维依赖拆成 chunk boundary。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L40 在 RL 与对齐主线中的位置。
2. 读 [lecture.md](lecture.md)：从 GAE 公式、last_value、chunk boundary、SLiME 生产路径讲到测试边界。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、测试、SLiME vanilla/chunked GAE 和 returns 读源码。
4. 做 quiz：确认 `next_value`、`next_adv`、boundary state、余数 chunk 和性能结论边界。
5. 做 patch：实现 `gae_naive` 和 `gae_chunked_parallel`。
6. 跑 patch-test：本讲没有 dedicated smoke/drill target，CPU 侧验收以 patch-test 为准。
7. 使用 [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) 记录一次数值或真实 GPU 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | 第 4 章：RL 与对齐 / PPO-GRPO 训练系统 |
| 解决什么问题 | 长上下文 RL 中 GAE 沿时间维反向递推，如何在保持数值等价的前提下暴露 chunk 并行结构 |
| 连接哪些源码 | `labs/l34.5_gae_chunked_parallel/patch/reference/gae_chunk.py`, `labs/l34.5_gae_chunked_parallel/patch/tests/test_patch.py`, `github_repo/slime/slime/utils/ppo_utils.py` |
| lab 检验什么 | naive GAE、chunked GAE、last_value、余数 chunk、single-chunk fallback 和 batched shape |

## 你会学到什么

- 写出 `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)` 和 `A_t = delta_t + gamma * lambda * A_{t+1}` 的代码状态。
- 区分 `next_value` 和 `next_adv`，说明 `last_value` 如何进入链尾。
- 解释 chunk `[start,end)` 的 boundary advantage 和 boundary value 如何传给左侧 chunk。
- 对照 SLiME 的 `vanilla_gae`、`chunked_gae`、pad/slice 和 returns 路径。
- 说明 CPU patch-test、源码映射和真实 GPU 加速比分别需要哪些证据。

## Patch 闭环

```bash
cat labs/l34.5_gae_chunked_parallel/patch/task.md
$EDITOR labs/l34.5_gae_chunked_parallel/patch/starter/gae_chunk.py
make patch-test M=l34.5_gae_chunked_parallel
```

参考实现验收：

```bash
IMPL=reference make patch-test M=l34.5_gae_chunked_parallel
```

本讲没有 dedicated smoke/drill target；真实长上下文加速需要在 GPU kernel 或 SLiME 训练 step 中单独计时。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 GAE 方向、last_value、chunk boundary、余数 chunk 和性能结论 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch 与 SLiME 源码主路径 |
| [outputs/rl_rollout_template.md](outputs/rl_rollout_template.md) | 记录数值等价、源码定位、真实 GPU 条件和结论 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L40 后进入 [L41 · Multimodal Capstone](../l35_multimodal_capstone/README.md)。
