# L01 课后产物

本目录存放环境课后可以直接复用的排查材料。它们服务于真实项目接手和后续 lab 排障，不是提交作业的格式。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | import、CUDA、NCCL、torchrun、环境漂移的排查顺序 |
| `source_reading_card.md` | 快速回忆本讲源码主路径和每个文件的结论 |
| `env_report_template.md` | 跑完 smoke 后记录命令、配置、artifact、字段判断和证据边界 |

建议每次换机器、换容器、换 Conda 环境、换 PyTorch wheel 或换集群启动方式时，都填一次 `env_report_template.md`。后续训练或 serving 出现异常时，先拿最近一次通过的环境报告做对照。
