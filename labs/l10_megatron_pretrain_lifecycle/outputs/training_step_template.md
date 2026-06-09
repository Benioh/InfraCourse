# L11 训练生命周期复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- 命令：
- run id：
- git commit：
- PyTorch / Megatron 版本：
- train_step 实现：
- scheduler 实现：

## 配置

| 字段 | 值 | 判断 |
|---|---:|---|
| profile |  |  |
| hidden_size |  |  |
| num_layers |  |  |
| seq_length |  |  |
| micro_batch_size |  |  |
| train_steps |  |  |
| lr |  |  |
| min_lr |  |  |
| restart_steps |  |  |
| save_every |  |  |

## Metrics 抽查

| iteration | loss | lr | num_microbatches | skipped_iter | tokens | 判断 |
|---:|---:|---:|---:|---:|---:|---|
| 1 |  |  |  |  |  |  |
| 50 |  |  |  |  |  |  |
| last |  |  |  |  |  |  |

## Acceptance

| 字段 | 值 | 判断 |
|---|---:|---|
| head_loss |  |  |
| tail_loss_at_50 |  |  |
| loss_drop |  |  |
| loss_drop_required |  |  |
| restart_ok |  |  |
| final_loss |  |  |
| final_loss_ok |  |  |
| all_passed |  |  |

## Checkpoint

| artifact | 路径 | 检查结果 |
|---|---|---|
| checkpoint file |  |  |
| latest marker |  |  |
| config.resolved.yaml |  |  |
| metrics.jsonl |  |  |
| report.md |  |  |

## 源码对应

| 现象 | 源码位置 | 判断 |
|---|---|---|
| zero_grad 顺序 |  |  |
| loss 平均 |  |  |
| optimizer skip |  |  |
| scheduler step |  |  |
| checkpoint marker |  |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 进入真实 Megatron 前还缺什么：
- 下一步动作：
