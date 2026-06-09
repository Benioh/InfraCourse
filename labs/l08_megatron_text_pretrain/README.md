# L09 · Megatron 文本预训练：训练 step 与 LR scheduler

这一讲把 L08 产出的 `--data-path <prefix>` 接到 Megatron-shaped 训练闭环。学生需要看清一条训练 step 如何从 data iterator 进入 forward/backward，再经过 optimizer、LR scheduler、training log 和 checkpoint。Patch 只验收一个最小组件：实现 `CosineWithRestartsLR`，用它练习训练框架里 scheduler 的状态、边界、param group 同步和恢复风险。

## 学习路线

建议按下面顺序走，先理解训练 loop 和指标闭环，再写 scheduler patch。

1. 读 [system_map.md](system_map.md)：确认 L09 在训练数据、Megatron 预训练和长上下文路线之间的位置。
2. 读 [lecture.md](lecture.md)：理解训练 step、有效 batch、LR 调度、日志、checkpoint 和 fallback 验证。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 MiniInfra、Megatron 源码、patch reference 和脚本主路径读。
4. 本讲没有 notebook；用 `scripts/run_train.py` 和 `scripts/parse_megatron_log.py` 观察 artifact。
5. 做 quiz：确认 scheduler 计数、restart 边界、param group 同步和日志字段。
6. 做 patch：实现 `CosineWithRestartsLR`，并通过 7 个 CPU 测试。
7. 跑 drill：记录启动命令、fallback 原因、metrics、train.log 和 report。
8. 填写 [outputs/training_step_template.md](outputs/training_step_template.md)，沉淀一次训练 step 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems / Megatron pretraining |
| 它承接什么 | L08 的 IndexedDataset prefix 和 Megatron `--data-path` |
| 它解决什么问题 | 让学生读懂预训练 step 中 optimizer、scheduler、日志和 checkpoint 的边界 |
| 它连接哪些指标或证据 | `lr`、`lm_loss`、`grad_norm`、`tokens/sec`、`mfu`、`skipped_iter`、fallback reason、expected command |
| 它连接哪些源码 | MiniInfra `train_step`、Megatron `OptimizerParamScheduler`、Megatron `training.py`、本关 patch 和 drill 脚本 |
| lab 检验什么 | 自定义 cosine-with-restarts scheduler 的边界、公式、param group 同步和 `get_lr()` 合同 |

## 你会学到什么

- 画出 Megatron 预训练从 data-path 到 train step、日志和 checkpoint 的主路径。
- 区分 iteration-based schedule 和 sample-based schedule，并解释 effective batch 对 LR 计数的影响。
- 解释 scheduler 的输入、中间状态、输出、代价和恢复边界。
- 读懂 Megatron `OptimizerParamScheduler.get_lr()`、`step()` 和训练日志里的 learning rate 流向。
- 用 fallback drill 记录启动证据，避免把缺少真实 Megatron runtime 的本地验证误报成真实训练。

## Patch 闭环

```bash
cat labs/l08_megatron_text_pretrain/patch/task.md
$EDITOR labs/l08_megatron_text_pretrain/patch/starter/lr_scheduler.py
make patch-test M=l08_megatron_text_pretrain
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_initial_lr_is_max` | step 0 时 lr 等于 `max_lr` |
| `test_lr_decreases_within_segment` | 段内 lr 单调下降 |
| `test_lr_at_restart_step_is_max` | restart step 回到 `max_lr` |
| `test_lr_clamps_after_total` | 超过 total 后钳到 `min_lr` |
| `test_multiple_param_groups_synced` | 所有 optimizer param groups 同步 |
| `test_get_lr_matches_optimizer` | `get_lr()` 与 optimizer 中的 lr 一致 |
| `test_no_restarts_is_pure_cosine` | 空 restart 列表退化为单段 cosine |

## Drill 闭环

从仓库根目录运行：

```bash
(cd labs/l08_megatron_text_pretrain && bash scripts/train_4090.sh l09_local)
python labs/l08_megatron_text_pretrain/scripts/parse_megatron_log.py \
  runs/l08_megatron_text_pretrain/l09_local/train.log
python labs/l08_megatron_text_pretrain/scripts/parse_megatron_log.py --self-test
```

如果本机没有真实 Megatron 包或 L08 数据产物，drill 会写出 fallback artifact。它的价值是记录实际启动边界：缺了 JSONL、缺了 `.bin/.idx`、还是 Megatron 包不可导入。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 LR 曲线错位、loss 抖动、resume 不连续和训练启动证据不足 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习训练 step、scheduler、日志和 patch 的源码主路径 |
| [outputs/training_step_template.md](outputs/training_step_template.md) | 记录一次训练运行的配置、指标、日志字段、fallback 和结论 |

## 进入下一讲

`make patch-test M=l08_megatron_text_pretrain` 通过，并完成一次 `train_4090.sh` 或 log parser 复盘后，进入 [L10 长上下文 CP](../l09_long_context_cp/README.md)。下一讲会从训练 loop 进入长上下文 attention 的计算和通信边界。
