# L04.8 · Megatron Pretrain Lifecycle：把 train_step 接回真实预训练

> 本关补上 L04 的框架主线：在 `mini_infra/megatron/training/training.py` 里手写 `train_step`
> 切片，并通过 `scripts/run_lifecycle.py` 把它驱动起来——你能第一次"看见"loss 真正下降。

L04 让你写了 `CosineWithRestartsLR`。L04.8 让你把 LR / optimizer / forward-backward / scheduler
/ metrics 串起来，并在一个 125M Llama-style 模型上跑 200–1000 步真实预训练（或者在
本地 fallback 模式下走通同构生命周期）。

## 闭环

```bash
# 1. 读任务、写 patch
cat labs/l10_megatron_pretrain_lifecycle/patch/task.md
$EDITOR labs/l10_megatron_pretrain_lifecycle/patch/starter/train_step.py
make patch-test M=l10_megatron_pretrain_lifecycle

# 2. 用 patch 驱动一次完整 lifecycle（4090/CPU fallback 都能跑）
bash labs/l10_megatron_pretrain_lifecycle/scripts/launch_pretrain.sh

# 3. 看 runs/<mission>/<run-id>/ 下的 metrics.jsonl + report.md
```

## 测试覆盖（patch 层）

| 测试 | 验证 |
|---|---|
| `test_train_step_call_order_and_metrics` | zero_grad → forward_backward → optimizer.step → scheduler.step 顺序，metrics 完整 |
| `test_scheduler_not_stepped_when_optimizer_skips` | optimizer 返回 False 时 scheduler 不前进，`skipped_iter=1` |
| `test_accepts_optimizer_step_none_as_success` | optimizer 返回 None 视为成功 |
| `test_rejects_missing_loss` | 没有 loss/losses 抛 `TrainStepError` |
| `test_rejects_empty_microbatch_losses` | 空 microbatch 列表抛错 |

## Lifecycle 测试覆盖（scripts 层）

`scripts/run_lifecycle.py` 在 mock 数据上驱动 200 步训练，并检查：

- 训练 loss 在前 50 步内严格下降（≥ 0.3 的下降幅度）
- LR 在指定 step 触发 warm restart 后回到 max_lr
- checkpoint 写出后 latest_checkpointed_iteration.txt 与文件一致
- metrics.jsonl 每行包含 `iteration / loss / lr / num_microbatches / skipped_iter`

`acceptance` 字段会写到 `runs/.../report.md` 里，作为口试证据。

## Configs

| 配置 | 适用硬件 | 说明 |
|---|---|---|
| `configs/cpu_smoke.yaml` | CPU only | 4-layer 8M 参数，200 步，跑得通就行 |
| `configs/4090_debug.yaml` | 单卡 4090 | 125M Llama-style，500 步，验证 loss 单调 |
| `configs/h200_125m.yaml` | 8×H200 | 125M Llama-style + Megatron `pretrain_gpt.py` 真实路径，1000 步 acceptance loss < 6.5 |

## 调试工单

见 `tickets/INDEX.md`。建议至少做 `mgt_lifecycle_train_loss_nan` 和
`mgt_lifecycle_skip_step_storm`。

## 卡住怎么办

1. `make patch-hint M=l10_megatron_pretrain_lifecycle` 看 TODO 列表。
2. 看 `mini_infra/megatron/training/training.py` 同构骨架，对照 reference。
3. `make patch-show-solution M=l10_megatron_pretrain_lifecycle`。

## 进入下一关

`make patch-test` 全绿 + `bash scripts/launch_pretrain.sh` 跑通后，
进入 [L05 Megatron Scale Optimization](../l11_megatron_scale_optimization/README.md)。
