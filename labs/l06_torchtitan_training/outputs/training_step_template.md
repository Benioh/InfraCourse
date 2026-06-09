# L07 Activation Checkpoint 复盘模板

## 1. Run 信息

- 日期：
- git commit：
- 命令：
- 环境：CPU / GPU
- PyTorch：
- CUDA：
- policy：
- 模型 / shape：

## 2. 验证目标

| 项 | 内容 |
|---|---|
| 目标 |  |
| 比较对象 | 裸模型 / no checkpoint |
| 成功标准 |  |
| 已知边界 |  |

## 3. 功能结果

| 指标 | 值 | 判断 |
|---|---|---|
| output max diff |  |  |
| input grad max diff |  |  |
| param grad max diff |  |  |
| wrapped children |  |  |
| no-op policy result |  |  |

## 4. 显存和时间

| 指标 | base | wrapped | 判断 |
|---|---:|---:|---|
| peak memory |  |  |  |
| step time |  |  |  |
| forward count estimate |  |  |  |

如果 CUDA 测试 skip：

```text
GPU peak memory 未验证；当前结果只证明 CPU 数值语义。
```

## 5. 源码对应

| 机制 | 源码位置 | 判断 |
|---|---|---|
| wrapper forward | `patch/reference/selective_ckpt.py` L18-L19 |  |
| child replacement | `patch/reference/selective_ckpt.py` L22-L26 |  |
| attention policy | `patch/reference/selective_ckpt.py` L29-L31 |  |
| TorchTitan apply_ac | `activation_checkpoint.py` L204-L255 |  |
| parallelize order | `parallelize.py` L80-L102 |  |

## 6. 结论

- 本次能证明：
- 本次不能证明：
- 如果失败，最小复现命令：
- 下一步要改的 policy、模型或实验：
