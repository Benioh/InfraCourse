# 课后产物：L34 CUDA Graph Cache + Memory Savor

本目录存放本讲课后可复用的排查材料。它们用于记录一次 graph cache、pause/resume 或 co-locate 阶段切换的证据。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 排查 cache miss、地址变化、paused bytes、resume 失败和 co-locate OOM |
| `source_reading_card.md` | 快速回忆 patch、Megatron CUDA Graph 和 SLiME memory utility 主路径 |
| `rl_rollout_template.md` | 记录一次 CPU patch、notebook 或 GPU graph 实验复盘 |

每次复盘至少写清命令、输入 shape、capture/replay 计数、paused bytes、GPU 地址证据、timing 和结论边界。
