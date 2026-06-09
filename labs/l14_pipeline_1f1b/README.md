# L15 · Pipeline Parallel 1F1B：把大模型训练 step 切成流水线

这一讲看 Pipeline Parallelism（PP）的调度控制面：模型被切成多个 stage 后，microbatch 怎样进入 forward，梯度怎样按反向依赖回流，1F1B 怎样减少 GPipe 式全前向再全反向带来的激活驻留。训练吞吐、显存峰值、pipeline bubble 和通信等待都和这个 schedule 有关。

本关 patch 只实现一个教学版 `make_1f1b_schedule(num_stages, num_microbatches)` 和 `bubble_count(num_stages)`。它不运行真实模型，也不发 P2P tensor；讲授重点是把 warmup、steady 1F1B、cooldown、microbatch 依赖和 Megatron 非 interleaved schedule 的边界讲清楚。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L15 在训练并行主线中的位置。
2. 读 [lecture.md](lecture.md)：理解 pipeline stage、microbatch、bubble、activation 驻留和 1F1B 三段式。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra、Megatron non-interleaved schedule 主路径读源码。
4. 跑 notebook：本讲没有强依赖 notebook，优先跑 CPU smoke。
5. 做 quiz：确认 warmup/cooldown、bubble、microbatch 数和 interleaved PP 的边界。
6. 做 patch：实现 1F1B schedule generator 并通过 CPU 测试。
7. 跑 drill：用 mock 时间模型输出 bubble ratio 和 schedule artifact。
8. 填写 [outputs/training_step_template.md](outputs/training_step_template.md)，沉淀一次 PP 调度复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems / model parallel training |
| 它承接什么 | L13 的分片状态视角和 L14 的跨 rank token 通信视角 |
| 它解决什么问题 | 模型层数太多或单 stage 显存不足时，怎样把一个训练 step 切成 pipeline stage 与 microbatch |
| 它连接哪些指标 | bubble count、bubble ratio、stage idle time、activation memory、step time、P2P wait |
| 它连接哪些源码 | patch `pp_schedule.py`、MiniInfra schedules、Megatron `forward_backward_pipelining_without_interleaving` |
| lab 检验什么 | warmup、steady 1F1B、cooldown、microbatch 顺序和 bubble 公式 |

## 你会学到什么

- 解释 PP 为什么需要 microbatch，以及 microbatch 数怎样影响 bubble ratio。
- 画出 non-interleaved 1F1B 的 warmup、steady、cooldown 三段。
- 区分 GPipe、1F1B、interleaved 1F1B 的激活驻留和调度代价。
- 读懂 Megatron 根据 PP size / virtual PP size 选择 forward-backward 函数的入口。
- 在 Megatron non-interleaved schedule 中找到 warmup forward、steady forward/backward、cooldown backward 和 gradient finalize。
- 用 drill 解释 bubble ratio 高、stage 空闲或 microbatch 配置不合理的问题。

## Patch 闭环

```bash
cat labs/l14_pipeline_1f1b/patch/task.md
$EDITOR labs/l14_pipeline_1f1b/patch/starter/pp_schedule.py
make patch-test M=l14_pipeline_1f1b
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_warmup_lengths` | stage `s` 的 warmup forward 数是 `D - s - 1` |
| `test_steady_alternates_F_B` | steady 阶段 forward/backward 严格交替 |
| `test_cooldown_lengths` | cooldown backward 数和 warmup 对称 |
| `test_each_microbatch_has_one_F_one_B_per_stage` | 每个 stage 上每个 microbatch 都有一次 F 和一次 B |
| `test_microbatch_order_per_stage` | 同一 stage 上 microbatch 顺序递增 |
| `test_no_microbatch_backward_before_its_forward` | 同一 microbatch 的 backward 不能早于 forward |
| `test_bubble_count_2_times_pp_minus_1` | 标准 1F1B bubble 为 `2 * (D - 1)` |
| `test_invalid_num_microbatches_raises` | microbatch 数不足时显式拒绝 |

## Drill 闭环

```bash
bash labs/l14_pipeline_1f1b/scripts/run_pp_smoke.sh l15_pp_smoke
```

CPU smoke 默认使用 4 个 stage、8 个 microbatch、forward time 1.0、backward time 2.0。脚本会写入 `runs/l14_pipeline_1f1b/<run-id>/`，其中 `artifacts/pp_summary.json` 包含 `bubble_count`、`bubble_ratio`、`schedule_head` 和 `schedule_last`。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 warmup 偏移、steady 顺序错误、bubble ratio 偏高和 P2P 等待 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速回忆 patch、MiniInfra 和 Megatron PP schedule 主路径 |
| [outputs/training_step_template.md](outputs/training_step_template.md) | 记录一次 PP smoke、benchmark 或真实训练 step 的配置、指标和判断 |

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 4 stages x 8 microbatches 的本地 smoke |
| `configs/h200_pp4.yaml` | 4-way PP 的 Megatron 命令模板 |
| `configs/h200_pp8.yaml` | 8-way PP 的 Megatron 命令模板 |

## 进入下一讲

`make patch-test M=l14_pipeline_1f1b` 通过，并完成一次 PP smoke 复盘后，进入 [L16 Megatron Parallel Checkpoint](../l15_megatron_parallel_checkpoint/README.md)。
