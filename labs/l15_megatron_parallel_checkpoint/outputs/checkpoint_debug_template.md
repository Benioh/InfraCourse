# L16 Checkpoint Debug 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- 配置文件：
- run id：
- git commit：
- checkpoint format：
- 实现来源：starter / reference / Megatron

## 保存与加载拓扑

| 项 | 保存时 | 加载时 |
|---|---:|---:|
| tensor parallel (`tp`) |  |  |
| pipeline parallel (`pp`) |  |  |
| data parallel (`dp`) |  |  |
| expert parallel (`ep`) |  |  |
| context parallel (`cp`) |  |  |
| world size |  |  |

## Checkpoint 入口

| 项 | 值 |
|---|---|
| checkpoint dir |  |
| latest marker path |  |
| latest marker content |  |
| checkpoint payload path |  |
| payload format |  |
| payload iteration |  |

## 状态完整性

| 状态 | 是否存在 | 证据 / 路径 | 判断 |
|---|---|---|---|
| model_state |  |  |  |
| optimizer_state |  |  |  |
| scheduler_state / opt_param_scheduler |  |  |  |
| parallel_state / args |  |  |  |
| RNG state |  |  |  |
| dataloader / consumed samples |  |  |  |

## Load 结果

| Case | strict 结果 | non-strict warnings | 判断 |
|---|---|---|---|
| same topology |  |  |  |
| TP changed |  |  |  |
| PP changed |  |  |  |
| EP changed |  |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| marker 解析 | `patch/reference/checkpointing.py` L48-L59 或 Megatron `read_metadata` |  |
| payload 保存 | `patch/reference/checkpointing.py` L28-L45 或 Megatron `generate_state_dict` |  |
| parallel_state mismatch | `patch/reference/checkpointing.py` L72-L80 |  |
| optimizer shard metadata | `distrib_optimizer.py` L1336-L1456 |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 下一步要改的配置、代码、转换流程或恢复策略：
