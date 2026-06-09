# L15 Debug Checklist：Pipeline Parallel 1F1B

## 1. 先固定现场

- 记录命令、配置文件、run id、git commit、Python 环境、GPU/节点数和并行配置。
- 记录 `num_stages`、`num_microbatches`、forward/backward time 模型、global batch 和 microbatch size。
- 保存 `metrics.jsonl`、`artifacts/pp_summary.json`、stdout/stderr 和 Megatron 命令模板。
- 明确当前是 patch-test、CPU smoke、Megatron dryrun，还是真实多机训练。

## 2. 先看最小不变量

| 检查项 | 期望 | 异常时先看 |
|---|---|---|
| warmup | stage `s` 有 `D - s - 1` 个前置 forward | stage index 是否从 0 开始 |
| steady | 进入稳定段后严格 F/B 交替 | forward/backward 游标是否混用 |
| cooldown | backward 数和 warmup 对称 | cooldown 是否漏掉最后几个 microbatch |
| microbatch 顺序 | 同一 stage 上 F 和 B 的 index 都递增 | 是否用反向顺序生成本地 B |
| bubble count | `2 * (D - 1)` | 是否把 bubble ratio 和 bubble count 混在一起 |
| total ops | `D * N * 2` | 是否漏掉某个 stage 或某个 microbatch |

## 3. 再判断系统层问题

| 现象 | 可能原因 | 下一步 |
|---|---|---|
| patch-test 失败 | timeline 顺序或输入校验错误 | 先读 reference，再对照失败测试的断言 |
| `bubble_ratio` 高 | microbatch 数太少或 schedule 生成错误 | 比较 `schedule_head`、`schedule_last` 和理论 warmup |
| 真实训练 P2P wait 高 | stage 间 tensor 通信慢、stage 切分不均或网络拓扑不合适 | 看 Megatron timers 和通信日志 |
| 显存峰值高 | 激活驻留窗口大、checkpoint 配置不足或 deallocate 未生效 | 看 activation checkpoint 和 schedule 阶段 |
| step time 抖动 | microbatch 耗时不均、数据 shape 变化或某个 stage 变慢 | 分 stage 记录 forward/backward 时间 |

## 4. 沿源码主路径复查

1. `labs/l14_pipeline_1f1b/patch/starter/pp_schedule.py`：学生实现是否按 warmup、steady、cooldown 三段生成。
2. `labs/l14_pipeline_1f1b/patch/reference/pp_schedule.py`：参考实现的游标和边界。
3. `labs/l14_pipeline_1f1b/patch/tests/test_patch.py`：当前失败断言对应哪个不变量。
4. `labs/l14_pipeline_1f1b/scripts/run_pp_smoke.py`：bubble ratio 和 artifact 是否来自正确配置。
5. `mini_infra/megatron/core/pipeline_parallel/schedules.py`：教学事件流是否符合 forward 向后、backward 向前。
6. `github_repo/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`：真实 non-interleaved schedule 的 warmup、steady、cooldown 循环。

## 5. 结束条件

- 最小复现命令可以稳定重跑。
- patch-test、smoke 或真实训练日志已经落盘。
- 能指出异常对应 warmup、steady、cooldown、P2P 通信还是 gradient finalize。
- 结论写入 `training_step_template.md`，并明确下一步是改 schedule、改 microbatch、改 stage 切分，还是看真实通信。
