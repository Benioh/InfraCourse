# L11 · Megatron 预训练生命周期：把 train_step 接回训练闭环

这一讲把 L09 的 scheduler 组件和 L10 的长上下文系统视角接回 Megatron-shaped 预训练生命周期。学生要写的 patch 是 `train_step`：清梯度、执行 forward/backward、调用 optimizer、按 update success 决定 scheduler 是否推进，并返回可写入日志的 metrics。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L11 在 Megatron 预训练生命周期中的位置。
2. 读 [lecture.md](lecture.md)：理解任务入口、通用训练循环、microbatch loss、optimizer skip、scheduler、metrics 和 checkpoint。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra、run_lifecycle 和真实 Megatron `training.py` 读源码。
4. 跑 notebook：`notebooks/n06_activation_recompute.ipynb`，观察 activation recompute 对训练 step 的影响。
5. 做 quiz：确认调用顺序、loss 汇总、skip 语义、metrics 和 checkpoint 证据。
6. 做 patch：实现 `train_step` 并通过 5 个 CPU 测试。
7. 跑 drill：用 `scripts/run_lifecycle.py` 产出 `metrics.jsonl`、`acceptance.json`、checkpoint marker 和 report。
8. 填写 [outputs/training_step_template.md](outputs/training_step_template.md)，沉淀一次生命周期复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems / Megatron lifecycle |
| 它承接什么 | L09 的 LR scheduler 组件、L10 的训练系统证据意识 |
| 它解决什么问题 | 把单步训练合同写成可测试边界，并让日志和 checkpoint 能解释训练行为 |
| 它连接哪些指标或证据 | `iteration`、`loss`、`num_microbatches`、`skipped_iter`、`lr`、`grad_norm`、`tokens`、checkpoint marker |
| 它连接哪些源码 | patch train_step、MiniInfra `training.py`、run_lifecycle、checkpointing、Megatron `training.py` |
| lab 检验什么 | 调用顺序、microbatch loss 平均、optimizer skip、scheduler step、异常输入 |

## Patch 闭环

```bash
cat labs/l10_megatron_pretrain_lifecycle/patch/task.md
$EDITOR labs/l10_megatron_pretrain_lifecycle/patch/starter/train_step.py
make patch-test M=l10_megatron_pretrain_lifecycle
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_train_step_call_order_and_metrics` | `zero_grad -> forward_backward -> optimizer.step -> scheduler.step` 和 metrics |
| `test_scheduler_not_stepped_when_optimizer_skips` | optimizer skip 时 scheduler 不推进 |
| `test_accepts_optimizer_step_none_as_success` | optimizer 返回 `None` 视为成功 |
| `test_rejects_missing_loss` | 缺少 loss 字段时抛错 |
| `test_rejects_empty_microbatch_losses` | 空 microbatch loss 列表时抛错 |

## Drill 闭环

```bash
python labs/l10_megatron_pretrain_lifecycle/scripts/run_lifecycle.py \
  --config configs/cpu_smoke.yaml --run-id l11_validation
```

drill 会用 tiny decoder 和 synthetic token 跑一个 CPU lifecycle，写出 `metrics.jsonl`、`artifacts/acceptance.json`、checkpoint marker 和 `report.md`。如果配置带真实 Megatron 命令模板，脚本会把命令写入 `artifacts/real_megatron_command.sh`。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 loss 不降、skip storm、LR 不动、checkpoint 缺失和 lifecycle 证据不足 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 train_step、MiniInfra lifecycle、checkpoint 和 Megatron 源码主路径 |
| [outputs/training_step_template.md](outputs/training_step_template.md) | 记录一次训练 step / lifecycle 的配置、指标、artifact 和判断 |

## 进入下一讲

`make patch-test M=l10_megatron_pretrain_lifecycle` 通过，并完成一次 `run_lifecycle.py` 复盘后，进入 [L12 Megatron Scale Optimization](../l11_megatron_scale_optimization/README.md)。
