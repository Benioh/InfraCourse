# 课后产物：L32 Train-Infer Mismatch

本目录存放本讲课后可复用的排查材料。它们用于记录一次 mismatch 现象、源码判断和下一步动作，不是作业提交格式。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序排查 K3 KL 上升、ratio 长尾、veto 命中和 batch norm 漂移 |
| `source_reading_card.md` | 快速回忆 patch 与 SLiME `mis.py` 的源码主路径 |
| `rl_rollout_template.md` | 记录一次 notebook、patch-test 或真实 RL 运行的复盘 |

每次复盘至少写清命令、配置、输入规模、关键指标、源码位置和结论边界。缺少这些证据时，只能说“观察到现象”，不能写成生产判断。
