# L05.8.5 · Crash-resume：杀掉训练后还能继续

> 本关只做一件事：**写一个 crash-safe checkpoint 写入 + resume 流程**——
> 在训练 step N 中途 SIGKILL 进程，重启之后 step N+1 的 loss 必须与不中断的
> baseline run 在 1% 之内一致。

之前 L05.8 只测试 save/load 的*文件形状*。L05.8.5 真正演练"训练崩溃→重启→收敛"，
是工业训练的硬门槛。

## 闭环

```bash
cat labs/l16_resume_after_crash/patch/task.md
$EDITOR labs/l16_resume_after_crash/patch/starter/crash_safe.py
make patch-test M=l16_resume_after_crash

bash labs/l16_resume_after_crash/scripts/run_crash_drill.sh
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_atomic_save_writes_tmp_then_rename` | 文件先写 `.tmp` 再 rename，避免半写文件被读 |
| `test_partial_tmp_only_is_discarded` | 只有 `.tmp` 存在时 load 走更早的 commit |
| `test_resume_restores_step_rng_optimizer` | step / torch RNG / optimizer state 全恢复 |
| `test_bit_exact_loss_continuation` | crash + resume 后续 loss 与不中断 baseline 一致 |
| `test_save_skips_when_step_unchanged` | 同 step 重复 save 是 idempotent |
| `test_load_when_no_checkpoint_returns_step_zero` | 全新启动返回 step=0 |

## Drill

`scripts/run_crash_drill.py`：
1. 不中断 baseline 跑 100 步，记录 loss[100]
2. 同 seed 跑 50 步，模拟 crash（直接退出），落盘最新 checkpoint
3. 用 patch 的 `load_latest` 恢复，再跑 50 步，记录 loss[100]
4. 比对：`|loss_baseline - loss_resumed| / loss_baseline < 1%`

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | tiny LM，CPU 演练完整 baseline / resume / 比对 |
| `configs/4090_125m.yaml` | 真正在 125M 模型上演练；需 GPU |

## 进入下一关

通过后回到 [L06 多模态数据](../l17_megatron_multimodal_data/README.md)。
