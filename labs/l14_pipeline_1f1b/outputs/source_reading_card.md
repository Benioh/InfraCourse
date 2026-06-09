# L15 Source Reading Card：Pipeline Parallel 1F1B

## 主路径

1. `labs/l14_pipeline_1f1b/patch/starter/pp_schedule.py`：最小 timeline 生成合同。
2. `labs/l14_pipeline_1f1b/patch/reference/pp_schedule.py`：warmup、steady、cooldown 的参考实现。
3. `labs/l14_pipeline_1f1b/patch/tests/test_patch.py`：8 条行为不变量。
4. `labs/l14_pipeline_1f1b/scripts/run_pp_smoke.py`：mock 时间模型、acceptance 和 artifact。
5. `mini_infra/megatron/core/pipeline_parallel/schedules.py`：教学版 pipeline event。
6. `github_repo/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`：Megatron non-interleaved 1F1B 主路径。

## 关键行

| 文件 | 行 | 读完要得到的结论 |
|---|---|---|
| patch starter | L6-L14 | 学生要生成 per-stage timeline，不处理真实 tensor |
| patch reference | L19-L31 | 三段式直接对应 warmup、steady、cooldown |
| patch tests | L20-L90 | 测试把顺序、数量和非法输入都钉住 |
| run_pp_smoke | L70-L85 | drill 用配置和 mock 时间模型算 bubble ratio |
| MiniInfra schedules | L31-L57 | forward 事件向后走，backward 事件向前走 |
| Megatron schedules | L2166-L2169 | 真实 schedule 也先计算 warmup 和 remaining |
| Megatron schedules | L2223-L2364 | warmup、steady、cooldown 三个循环驱动 P2P 和 backward |

## 先跳过

- interleaved pipeline：等 non-interleaved 主线读通后再看。
- multi-module pipeline：本讲不展开 VLM 多模块 pipeline。
- cuda graph / activation offload：这些是性能和内存分支，不改变 1F1B 顺序。
- 具体模型层 forward：L15 只关心 schedule 怎样调用 forward/backward。

## 自检

- 我能否解释 stage `s` 的 warmup 为什么是 `D - s - 1`？
- 我能否指出 patch 的 steady 循环在 Megatron 里对应哪些 send/recv 调用？
- 我能否说明 `deallocate_output_tensor` 为什么和激活内存有关？
- 我能否用 `pp_summary.json` 判断 schedule 是否符合理论 bubble？
