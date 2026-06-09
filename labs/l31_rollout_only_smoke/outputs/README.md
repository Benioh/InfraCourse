# 课后产物：L35 Async Rollout Pool

本目录保存三份可复用材料，用来复盘 rollout-only smoke、async pool 行为合同和真实 endpoint 排障。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 排查 rollout 慢、顺序错位、server queue 积压、reward-ready 缺失和异常吞掉 |
| `source_reading_card.md` | 快速回忆本地 smoke、patch、vLLM/SGLang 和 SLiME 主路径 |
| `rl_rollout_template.md` | 记录一次 rollout-only run 的命令、配置、样本、指标和下一步 |

使用这些材料时，先保存 `rollouts.jsonl`、`metrics.jsonl`、`rl.log` 和 `report.md`。只有吞吐数字、没有样本和命令快照时，结论不能支撑后续调参。
