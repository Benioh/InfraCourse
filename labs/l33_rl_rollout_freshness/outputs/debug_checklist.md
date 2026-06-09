# L38 Debug Checklist：Rollout Freshness

## 1. 固定现场

- 记录命令、配置文件、run id、git commit、Python 环境、硬件和随机种子。
- 记录 server 列表、`max_staleness`、`update_rate`、`update_subset_size` 和 actor 更新周期。
- 保存 `metrics.jsonl`、`artifacts/rl_drill.json`、`report.md`、stdout/stderr 和真实 SLiME 日志。
- 明确这是 patch-test、CPU drill、H200 模板，还是真实 SLiME 训练。

## 2. 先看版本，再看 reward

| 问题 | 要查字段 | 可能结论 |
|---|---|---|
| 样本是否来自旧 policy | `actor_version`, `weight_version`, `staleness` | rollout engine 落后当前 actor |
| 是否某个 server 长期落后 | `server_id`, per-server freshness | 局部更新范围太小或固定子集被遗漏 |
| 是否准入过宽 | `max_staleness`, p99 staleness | 成功率高但样本偏旧 |
| 是否准入过严 | failures, no-fresh-server error | 同步频率或覆盖范围不足 |
| reward 是否可解释 | reward_mean, KL, staleness 分布 | reward 变化需要按版本区间分段看 |

## 3. 沿源码主路径复查

1. `labs/l33_rl_rollout_freshness/patch/reference/rollout_manager.py`：确认 freshness、round-robin 和局部更新合同。
2. `mini_infra/slime/ray/rollout.py`：确认 `meta_info` 是否写出版本字段。
3. `mini_infra/slime/ray/train_actor.py`：确认 actor train 后版本递增，并通过 `update_weights` 写回 rollout manager。
4. `github_repo/slime/train.py`：确认真实训练循环中 rollout、train、update_weights 的顺序。
5. `github_repo/slime/slime/ray/rollout.py`：确认可更新 engines、engine lock 和 rollout data 主路径。
6. `github_repo/slime/slime/utils/types.py`：确认 `Sample.weight_versions` 是否从 meta_info 收集。
7. `github_repo/slime/slime/backends/sglang_utils/sglang_engine.py`：确认 update payload 和 `get_weight_version()`。

## 4. 常见错误判断

- 只看 reward_mean，没有按 `staleness` 和 `server_id` 分段。
- 所有 server stale 时仍继续 generate，把旧 policy 样本混进训练。
- 局部更新时误改所有 server，掩盖 subset starvation。
- 只记录 actor step，没有记录 rollout engine 的 `weight_version`。
- 把 CPU drill 的 success rate 当成真实 GPU 吞吐。

## 5. 结束条件

- 问题能被一个最小命令复现。
- 版本字段、failure 和 staleness 分布已经落盘。
- 能指出版本证据在哪一层产生、丢失或被错误解释。
- 结论写进 `rl_rollout_template.md`，并包含下一步动作。
