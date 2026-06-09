# 课后产物：L33 DPO Loss

本目录存放本讲课后可复用的排查材料。它们用于记录一次 DPO 运行的配置、指标、源码判断和下一步动作。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 排查 completion mask、reference、beta、reward margin 和数值稳定性 |
| `source_reading_card.md` | 快速回忆 patch、pytest、smoke 和 RL logprob 对照源码 |
| `rl_rollout_template.md` | 记录一次 patch-test、smoke 或真实 DPO 训练复盘 |

每次复盘至少写清命令、配置、数据样本、mask 规则、beta、loss、reward margin 和 reference 冻结证据。
