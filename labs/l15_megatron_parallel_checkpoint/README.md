# L05.8 · Megatron Distributed Checkpoint：save / load / parallel-state 校验

> 本关补上 L05 的框架主线：实现 Megatron-shaped distributed checkpoint。
> 不仅要让 patch 通过，还要用 `scripts/run_checkpoint_drill.py` 演练一遍
> "save → 改 TP → load 失败 → strict=False 看 warning"的完整故事。

学完后你能解释为什么 checkpoint 文件存在不等于可 resume，以及 TP/PP/EP 配置变化
为什么必须先看 checkpoint metadata。

## 闭环

```bash
cat labs/l15_megatron_parallel_checkpoint/patch/task.md
$EDITOR labs/l15_megatron_parallel_checkpoint/patch/starter/checkpointing.py
make patch-test M=l15_megatron_parallel_checkpoint

# 演练 save/load/mismatch
bash labs/l15_megatron_parallel_checkpoint/scripts/run_drill.sh
```

## 测试覆盖（patch 层）

| 测试 | 验证 |
|---|---|
| `test_save_checkpoint_writes_payload_and_latest_marker` | 文件名 + latest marker + payload 含 format/iter |
| `test_load_checkpoint_roundtrip` | save → load 完全回环，无 warning |
| `test_strict_parallel_state_mismatch_raises` | strict=True 时 TP 不匹配抛 `CheckpointError` |
| `test_non_strict_parallel_state_mismatch_returns_warning` | strict=False 返回 warnings 列表 |
| `test_missing_latest_marker_raises` | 没有 latest marker 抛错 |

## Drill 演练（scripts 层）

`scripts/run_checkpoint_drill.py` 在临时目录里完成：

1. 用 `(tp=2, pp=1, dp=4)` 保存 iter_0000010
2. 用 `(tp=2, pp=1, dp=4)` strict load → 必须无 warning
3. 用 `(tp=4, pp=1, dp=2)` strict load → 必须抛 `CheckpointError`
4. 用 `(tp=4, pp=1, dp=2)` non-strict load → 必须有一条 `tp` warning

每步结果写入 `runs/<run-id>/artifacts/drill.json` 和 `metrics.jsonl`。

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 默认 drill：纯 dict checkpoint，CPU 即可 |
| `configs/h200_distributed.yaml` | 配 Megatron `pretrain_gpt.py --save / --load` 的真实命令模板 |

## 调试工单

见 `tickets/INDEX.md`。建议至少做 `ckpt_tp_mismatch` 与 `ckpt_resume_lr_jump`。

## 进入下一关

`make patch-test` + `bash scripts/run_drill.sh` 都通过后，进入
[L06 多模态数据](../l17_megatron_multimodal_data/README.md)。
