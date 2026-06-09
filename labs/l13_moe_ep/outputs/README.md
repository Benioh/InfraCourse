# L14 课后产物：MoE Router 与 Expert Parallelism

本目录存放 L14 课后可以继续复用的材料。它们面向 router collapse、capacity overflow、expert imbalance 和 all-to-all 慢，不是一次性作业模板。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按顺序定位 MoE 路由、容量、负载均衡和 EP 通信问题 |
| `source_reading_card.md` | 快速回忆 patch、MiniInfra、Megatron router 和 dispatcher 主路径 |
| `moe_routing_template.md` | 跑 drill、EP smoke 或真实 benchmark 后填写的指标复盘模板 |

建议每次跑完 patch、drill 或真实训练后，把 expert 数、top-k、capacity factor、EP/TP/DP 配置、tokens per expert、drop rate、aux loss 和 all-to-all 指标写进模板。缺少这些条件时，不要把局部现象写成生产结论。
