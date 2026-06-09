# L17 Debug Checklist：Crash Resume

## 1. 先固定现场

- 记录命令、配置文件、run id、git commit、Python 环境、硬件、seed 和 crash step。
- 保存 `artifacts/crash_drill.json`、`metrics.jsonl`、checkpoint 目录、latest marker 和 stdout/stderr。
- 明确当前是 patch-test、CPU drill、GPU smoke，还是真实训练恢复。

## 2. 先看文件提交边界

| 检查项 | 期望 | 异常时先看 |
|---|---|---|
| `.tmp` 文件 | load 前可以存在，load 后应清理 | `load_latest` 是否调用 cleanup |
| committed checkpoint | 至少有一个 `iter_*.pt` 正式文件 | `atomic_save` 是否执行 replace |
| latest marker | 指向已提交 checkpoint | marker 是否晚于 checkpoint 写入 |
| payload JSON | 可解析，包含 step/model/optimizer/RNG | 是否读到了半写文件 |
| idempotent save | 同 step 不增加正式文件数量 | 文件名是否固定为 step |

## 3. 再看恢复状态

| 状态 | 缺失风险 | 证据 |
|---|---|---|
| step | scheduler、日志和数据游标错位 | payload `step` |
| model_state | 参数位置错误 | payload `model_state` |
| optimizer_state | 下一步更新方向错位 | payload `optimizer_state` |
| rng_state | dropout、shuffle 或采样不一致 | payload `rng_state` |
| extra | loss history、数据游标等控制状态丢失 | payload `extra` |

## 4. 常见故障

| 现象 | 可能原因 | 下一步 |
|---|---|---|
| load 读到 PARTIAL | loader 没清理 `.tmp` | 先修 `_cleanup_partial` |
| marker 指向不存在文件 | marker 更新早于 checkpoint 提交 | 检查 `save_step` 顺序 |
| `max_rel_diff` 从 crash step 后变大 | model 或 optimizer 没恢复 | 对比 payload 中 `w` 和 `m` |
| 随机相关 loss 分叉 | RNG state 没保存或没恢复 | 检查 payload 和真实 RNG tracker |
| 同 step 多个文件 | save 文件名带随机后缀 | 固定 `iter_{step:07d}.pt` |
| CPU drill 通过但真实训练失败 | 多 rank、异步保存或文件系统语义不同 | 看 Megatron tracker、barrier 和 rank 日志 |

## 5. 沿源码主路径复查

1. `labs/l16_resume_after_crash/patch/reference/crash_safe.py`：tmp+fsync+replace、tmp cleanup 和 save_step 顺序。
2. `labs/l16_resume_after_crash/patch/tests/test_patch.py`：当前失败断言对应哪个 crash-resume 不变量。
3. `labs/l16_resume_after_crash/scripts/run_crash_drill.py`：baseline/resume loss 的生成和比较。
4. `mini_infra/megatron/training/checkpointing.py`：payload、latest marker 和 load 合同。
5. `github_repo/Megatron-LM/megatron/training/checkpointing.py`：真实 save finalization、tracker、optimizer 和 RNG load。

## 6. 结束条件

- 最小复现命令可以稳定重跑。
- checkpoint 目录中没有 dangling tmp。
- latest marker 指向 committed checkpoint。
- payload 包含 step、model、optimizer 和 RNG。
- baseline/resume 的 `max_rel_diff` 在阈值内，或超阈值原因已经定位。
- 结论写入 `checkpoint_debug_template.md`。
