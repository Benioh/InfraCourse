# L05.7 · Pipeline Parallel 1F1B 调度

> 本关只做一件事：**手写 1F1B（一前一反）pipeline schedule generator**——
> 给定 `num_stages`、`num_microbatches`，输出每个 stage 的 (Op, microbatch) 时间线，
> 与 Megatron / DeepSpeed 的 1F1B 一致。

之前课程没有 PP 任何形态。L05.7 把这个最重要的并行维度补上。

## 闭环

```bash
cat labs/l14_pipeline_1f1b/patch/task.md
$EDITOR labs/l14_pipeline_1f1b/patch/starter/pp_schedule.py
make patch-test M=l14_pipeline_1f1b
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_warmup_lengths` | stage s warmup = D - s - 1 |
| `test_steady_alternates_F_B` | steady 阶段 1 forward 1 backward 严格交替 |
| `test_cooldown_lengths` | stage s cooldown = D - s - 1 backwards |
| `test_each_microbatch_has_one_F_one_B_per_stage` | 数 forward/backward 总数正确 |
| `test_microbatch_order_per_stage` | 同 stage 上 forward 的 microbatch idx 单调递增 |
| `test_no_microbatch_backward_before_its_forward` | 同 stage 上 B(i) 必须在 F(i) 之后 |
| `test_bubble_count_2_times_pp_minus_1` | bubble = 2(D-1) |
| `test_invalid_num_microbatches_raises` | num_microbatches < num_stages 抛错 |

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 4 stages × 8 microbatches，与 Megatron 文档配图一致 |
| `configs/h200_pp4.yaml` | 4 stages × 64 microbatches，1B 模型 |
| `configs/h200_pp8.yaml` | 8 stages × 128 microbatches，13B 模型 |

## Drill

`scripts/run_pp_smoke.py` 把生成的 schedule 跑一遍 mock 计算（每 op 模拟固定耗时），
计算 throughput 与 bubble ratio，并和理论值比对。

## 进入下一关

通过后回到 [L05.8 distributed checkpoint](../l15_megatron_parallel_checkpoint/README.md)。
