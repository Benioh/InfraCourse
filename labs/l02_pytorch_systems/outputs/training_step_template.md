# Training Step 复盘模板

## Run 信息

- 日期：
- 机器 / GPU：
- git commit：
- 命令：
- 配置文件：
- run 目录：
- PyTorch / CUDA：

## 模型和输入

| 条件 | 数值 |
|---|---|
| batch size |  |
| sequence length |  |
| vocab size |  |
| hidden size |  |
| heads |  |
| layers |  |
| dtype / precision |  |
| activation checkpointing |  |
| optimizer |  |
| steps |  |

## 静态显存账本

| 项 | bytes | GB | 说明 |
|---|---|---|---|
| params |  |  |  |
| grads |  |  |  |
| optimizer state |  |  |  |
| static total |  |  |  |

## 运行时指标

| 指标 | p50 / mean | p95 / max | 判断 |
|---|---|---|---|
| `step_time_ms` |  |  |  |
| `tokens_per_sec` |  |  |  |
| `peak_memory_gb` |  |  |  |
| `dataloader_time_ms` |  |  |  |
| `forward_time_ms` |  |  |  |
| `backward_time_ms` |  |  |  |
| `optimizer_time_ms` |  |  |  |
| `grad_norm` |  |  |  |

## Profiler 证据

- trace 路径：
- trace 覆盖范围：
- warmup 是否完成：
- CPU 主要耗时：
- CUDA 主要耗时：
- 可疑同步点：

## 判断

- 静态账本能解释多少 peak memory：
- peak memory 中疑似 activation 或临时 buffer 的部分：
- step 时间主瓶颈：
- 改动一个变量的下一步实验：
- 预期影响的指标：

## 边界

- 本次结论适用的硬件和输入规模：
- 本次不能证明的内容：
- 需要进入下一讲或后续并行课程继续验证的问题：
