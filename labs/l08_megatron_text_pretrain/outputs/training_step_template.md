# L09 Megatron 训练 step 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- run id：
- git commit：
- Python / PyTorch / Megatron 版本：
- 是否真实 Megatron 训练：
- fallback 原因：

## 配置

| 字段 | 值 | 判断 |
|---|---:|---|
| model_size |  |  |
| seq_length |  |  |
| micro_batch_size |  |  |
| global_batch_size |  |  |
| TP |  |  |
| PP |  |  |
| precision |  |  |
| train_steps |  |  |
| data-path prefix |  |  |

## Scheduler

| 字段 | 值 | 判断 |
|---|---:|---|
| max_lr |  |  |
| min_lr |  |  |
| restart_steps |  |  |
| total_steps |  |  |
| step_count / num_steps |  |  |
| 当前 lr |  |  |
| 是否保存 scheduler state |  |  |

## Artifact 检查

| artifact | 路径 | 检查结果 |
|---|---|---|
| `config.resolved.yaml` |  |  |
| `command.sh` |  |  |
| `train.log` |  |  |
| `metrics.jsonl` |  |  |
| `report.md` |  |  |
| `fallback_reason.txt` |  |  |
| checkpoint |  |  |

## 日志字段

| 字段 | 值 | 判断 |
|---|---:|---|
| iteration |  |  |
| lm_loss |  |  |
| learning rate |  |  |
| grad_norm |  |  |
| consumed_samples |  |  |
| tokens/sec |  |  |
| MFU |  |  |
| skipped_iter |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| optimizer 成功后 scheduler 推进 |  |  |
| lr 写入 param groups |  |  |
| training_log 读取 canonical lr |  |  |
| checkpoint 保存 scheduler state |  |  |
| fallback 原因写入 artifact |  |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 若要进入真实 Megatron 训练，还缺什么：
- 若要支持 resume，还需要检查哪些状态：
- 下一步动作：
