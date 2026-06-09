# L16 · Megatron Parallel Checkpoint：把训练恢复合同保存完整

这一讲看分布式训练的恢复边界：checkpoint 文件存在，不等于训练可以无缝 resume。Megatron 训练会把模型权重、optimizer state、scheduler step、RNG、并行拓扑和 shard metadata 分散到不同结构里；TP/PP/EP/DP 配置变化后，旧 checkpoint 的切片含义也会变化。

本关 patch 实现一个教学版 Megatron-shaped checkpoint：保存 `model_state`、`optimizer_state`、`scheduler_state`、`parallel_state` 和 latest marker；加载时从 marker 找到最新 checkpoint，并根据 `expected_parallel_state` 做 strict / non-strict 校验。它不保存真实 sharded tensor，也不做跨 TP/PP reshard，重点是把 resume 合同讲清楚。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L16 在训练可靠性主线中的位置。
2. 读 [lecture.md](lecture.md)：理解 checkpoint payload、latest marker、parallel state、optimizer/scheduler state 和 Megatron metadata。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra、Megatron checkpointing 和 distributed optimizer 主路径读源码。
4. 跑 notebook：本讲没有强依赖 notebook，优先跑 checkpoint drill。
5. 做 quiz：确认 resume 合同、拓扑兼容性和 strict / non-strict load 行为。
6. 做 patch：实现 save/load 最小合同并通过 CPU 测试。
7. 跑 drill：演练同拓扑 load、TP mismatch strict 失败、non-strict warning。
8. 填写 [outputs/checkpoint_debug_template.md](outputs/checkpoint_debug_template.md)，沉淀一次 checkpoint 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training reliability / distributed checkpoint |
| 它承接什么 | L15 的 PP 调度与前面训练并行配置，说明这些拓扑信息怎样进入 checkpoint |
| 它解决什么问题 | 保存和恢复训练状态时，怎样判断 checkpoint 与当前并行拓扑是否兼容 |
| 它连接哪些指标 | latest iteration、load warnings、strict failure、optimizer shard format、resume step、LR continuity |
| 它连接哪些源码 | patch `checkpointing.py`、MiniInfra checkpointing、Megatron checkpointing、DistributedOptimizer sharded state dict |
| lab 检验什么 | payload 完整性、latest marker、format 校验、parallel_state strict / non-strict 语义 |

## 你会学到什么

- 解释完整 resume 为什么需要 model、optimizer、scheduler/global step 和 parallel state。
- 区分“加载权重做 finetune”和“恢复训练继续同一个 optimizer step”。
- 说明 latest marker 怎样避免用目录排序猜 checkpoint。
- 判断 TP/PP/EP/DP 变化为什么会影响 shard 形状和 optimizer state。
- 读懂 Megatron `get_checkpoint_name`、`read_metadata`、`generate_state_dict` 和 distributed optimizer sharding metadata 的主路径。
- 用 drill 和复盘模板定位 TP mismatch、missing marker、scheduler step 丢失和 optimizer shard 不兼容。

## Patch 闭环

```bash
cat labs/l15_megatron_parallel_checkpoint/patch/task.md
$EDITOR labs/l15_megatron_parallel_checkpoint/patch/starter/checkpointing.py
make patch-test M=l15_megatron_parallel_checkpoint
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_save_checkpoint_writes_payload_and_latest_marker` | 文件名、latest marker、format、iteration 和 scheduler state |
| `test_load_checkpoint_roundtrip` | 同拓扑 save/load 回环，无 warning |
| `test_strict_parallel_state_mismatch_raises` | strict 模式下 TP mismatch 抛 `CheckpointError` |
| `test_non_strict_parallel_state_mismatch_returns_warning` | non-strict 模式返回 warning |
| `test_missing_latest_marker_raises` | latest marker 缺失时显式失败 |

## Drill 闭环

```bash
IMPL=reference bash labs/l15_megatron_parallel_checkpoint/scripts/run_drill.sh l16_validation
```

CPU drill 默认保存 `(tp=2, pp=1, dp=4, ep=1)` 的 checkpoint，并验证：

| Case | 期望 |
|---|---|
| `same_topology` | strict load 通过，non-strict 无 warning |
| `tp_doubled` | strict load 抛错，non-strict 至少 1 条 warning |
| `ep_introduced` | strict load 抛错，non-strict 至少 1 条 warning |

结果写入 `runs/l15_megatron_parallel_checkpoint/<run-id>/artifacts/drill.json` 和 `metrics.jsonl`。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 missing marker、format mismatch、TP/PP/EP mismatch、LR 跳变和 optimizer shard 错误 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速回忆 patch、MiniInfra、Megatron checkpointing 与 optimizer sharding 主路径 |
| [outputs/checkpoint_debug_template.md](outputs/checkpoint_debug_template.md) | 记录一次 checkpoint save/load 或真实 resume 的证据和判断 |

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 默认 drill：纯 dict checkpoint，CPU 即可 |
| `configs/h200_distributed.yaml` | Megatron `pretrain_gpt.py --save / --load` 的真实命令模板 |

## 进入下一讲

`make patch-test M=l15_megatron_parallel_checkpoint` 通过，并完成一次 checkpoint drill 复盘后，进入 [L17 Crash Resume](../l16_resume_after_crash/README.md)。
