# L17 · Crash Resume：让训练在进程被杀后继续同一条轨迹

这一讲看训练可靠性里的 crash-resume 闭环：训练进程在 checkpoint 写入前后被杀，重启后应该加载最近一次完整提交的 checkpoint，恢复 step、model、optimizer、RNG 和附加状态，并让后续 loss 与不中断 baseline 对齐。这个问题比“文件能保存”更严格，因为半写文件、latest marker 错位、RNG 丢失或 optimizer state 丢失都会让恢复后的训练偏离原轨迹。

本关 patch 实现三个函数：`atomic_save(payload, target)`、`load_latest(checkpoint_dir)` 和 `save_step(...)`。它使用 JSON payload 模拟 checkpoint，不做真实 Megatron shard 存储；讲授重点是 crash-safe 写入、`.tmp` 清理、latest marker 更新顺序、idempotent save 和恢复后 loss 对齐。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L17 在训练可靠性主线中的位置。
2. 读 [lecture.md](lecture.md)：理解 crash-safe 写入、committed checkpoint、latest marker、RNG/optimizer 恢复和 baseline 对照。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、drill、MiniInfra 和 Megatron checkpointing 主路径读源码。
4. 跑 notebook：本讲没有强依赖 notebook，优先跑 crash drill。
5. 做 quiz：确认 atomic rename、`.tmp` 清理、RNG、optimizer 和 idempotent save。
6. 做 patch：实现 crash-safe checkpoint 最小合同并通过 CPU 测试。
7. 跑 drill：比对 baseline 与 crash+resume 的最终 loss。
8. 填写 [outputs/checkpoint_debug_template.md](outputs/checkpoint_debug_template.md)，沉淀一次 crash-resume 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training reliability / crash recovery |
| 它承接什么 | L16 的 checkpoint resume 合同，继续处理保存中途崩溃和恢复后数值连续性 |
| 它解决什么问题 | 进程在 checkpoint 写入中途被杀后，如何只加载完整提交的 checkpoint 并恢复训练轨迹 |
| 它连接哪些指标 | latest step、dangling tmp count、max relative loss diff、resume step、RNG/optimizer state restored |
| 它连接哪些源码 | patch `crash_safe.py`、crash drill、MiniInfra checkpointing、Megatron checkpoint finalization 和 load |
| lab 检验什么 | tmp+fsync+replace、tmp 清理、latest marker、state 恢复、idempotent save、loss 连续性 |

## 你会学到什么

- 解释为什么 checkpoint 写入要先写 `.tmp`，再 `fsync`，最后同文件系统 `os.replace`。
- 判断启动时遇到 dangling `.tmp` 应该如何处理。
- 说明 latest marker 何时更新，为什么必须晚于 checkpoint 文件提交。
- 解释 step、model、optimizer、RNG 和 extra state 对恢复后 loss 连续性的作用。
- 读懂 Megatron 保存 state、更新 tracker、加载 iteration、optimizer 和 RNG 的主路径。
- 用 crash drill 的 `max_rel_diff` 判断恢复后轨迹是否和 baseline 对齐。

## Patch 闭环

```bash
cat labs/l16_resume_after_crash/patch/task.md
$EDITOR labs/l16_resume_after_crash/patch/starter/crash_safe.py
make patch-test M=l16_resume_after_crash
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_atomic_save_writes_tmp_then_rename` | 写入后正式文件存在，`.tmp` 不残留 |
| `test_partial_tmp_only_is_discarded` | 只有 `.tmp` 的崩溃残留被删除，load 回到上一个 committed checkpoint |
| `test_resume_restores_step_rng_optimizer` | step、optimizer、RNG 和 extra state 都能恢复 |
| `test_save_skips_when_step_unchanged` | 同 step 重复 save 不产生重复 checkpoint 文件 |
| `test_load_when_no_checkpoint_returns_none` | 全新目录返回 `None` |
| `test_bit_exact_loss_continuation` | crash+resume 后 loss 与 baseline 对齐 |

## Drill 闭环

```bash
IMPL=reference bash labs/l16_resume_after_crash/scripts/run_crash_drill.sh l17_validation
```

CPU drill 默认用同一个 seed 生成 100 个梯度：

| 路径 | 行为 |
|---|---|
| baseline | 从 step 0 连续训练 100 步 |
| crash path | 训练 50 步，保存 checkpoint，加载后继续 50 步 |
| 验收 | `max_rel_diff <= 0.01` |

结果写入 `runs/l16_resume_after_crash/<run-id>/artifacts/crash_drill.json` 和 `metrics.jsonl`。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 half-written checkpoint、dangling tmp、marker 错位、RNG 丢失和 loss 偏离 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速回忆 patch、drill、MiniInfra 和 Megatron checkpoint 主路径 |
| [outputs/checkpoint_debug_template.md](outputs/checkpoint_debug_template.md) | 记录一次 crash-resume drill 或真实恢复事故的证据和判断 |

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 默认 drill：100 步 mock training，step 50 crash |
| `configs/4090_125m.yaml` | 在 125M 模型上演练，需要 GPU |

## 进入下一讲

`make patch-test M=l16_resume_after_crash` 通过，并完成一次 crash-resume drill 复盘后，进入 [L18 Megatron Multimodal Data](../l17_megatron_multimodal_data/README.md)。
