# L17 Crash Resume 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- 配置文件：
- run id：
- git commit：
- 实现来源：starter / reference / Megatron

## Crash 条件

| 项 | 值 |
|---|---|
| total steps |  |
| crash step |  |
| seed |  |
| baseline run |  |
| resumed run |  |
| acceptance threshold |  |

## Checkpoint 文件状态

| 项 | 值 / 路径 | 判断 |
|---|---|---|
| checkpoint dir |  |  |
| dangling `.tmp` count before load |  |  |
| dangling `.tmp` count after load |  |  |
| latest marker |  |  |
| latest marker content |  |  |
| committed checkpoint loaded |  |  |

## Payload 状态

| 状态 | 是否存在 | 证据 / 值 | 判断 |
|---|---|---|---|
| step |  |  |  |
| model_state |  |  |  |
| optimizer_state |  |  |  |
| rng_state |  |  |  |
| extra |  |  |  |

## Loss 对比

| 指标 | 数值 | 解释 |
|---|---:|---|
| final baseline loss |  |  |
| final resumed loss |  |  |
| max relative diff |  |  |
| first divergence step |  |  |
| accept |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| tmp 未清理 | `patch/reference/crash_safe.py` L27-L35 |  |
| marker 错位 | `patch/reference/crash_safe.py` L77-L79 |  |
| optimizer 未恢复 | `patch/tests/test_patch.py` L43-L57 |  |
| loss 分叉 | `scripts/run_crash_drill.py` L98-L125 |  |
| Megatron RNG load | `checkpointing.py` L1974-L2000 |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 下一步要改的配置、代码、保存策略或恢复策略：
