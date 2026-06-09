# L12 Debug Checklist：Bucketed Grad Sync 与 Scale Optimization

## 1. 先固定现场

- 记录命令、配置文件、git commit、Python 环境、硬件、world size、bucket size、TP/PP/DP/recompute 和随机种子。
- 保存 stdout/stderr、`metrics.jsonl`、`config.resolved.yaml`、`scale_table.json`、checkpoint marker 和 report。
- 明确这是 patch-test、scale drill、notebook smoke，还是真实 Megatron/TorchTitan 运行。

## 2. 判断问题在哪一层

| 层 | 要看什么 | 可能结论 |
|---|---|---|
| 输入 | batch、shape、dtype、world size、并行度 | 输入规模或并行配置已经偏离预期 |
| 状态 | buckets、grad、process group、optimizer state、checkpoint shards | 核心状态没有按机制推进 |
| 通信 | collective 次数、bucket size、overlap、reduce-scatter、all-gather | step time 被启动开销、带宽或同步等待拖慢 |
| 输出 | grad diff、loss、tokens/sec、MFU、peak memory、artifact | 结果无法支撑当前判断 |

## 3. 沿源码主路径复查

- `labs/l11_megatron_scale_optimization/patch/starter/bucketed_ddp.py`：学生需要补齐的最小同步合同。
- `labs/l11_megatron_scale_optimization/patch/reference/bucketed_ddp.py`：参考实现中的 bucket、flatten 和 all-reduce。
- `labs/l11_megatron_scale_optimization/patch/tests/conftest.py`：多进程 gloo harness。
- `labs/l11_megatron_scale_optimization/patch/tests/worker_cases.py`：PyTorch DDP 对照和边界 case。
- `github_repo/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py`：生产 DDP 的 bucket、buffer 和 overlap 入口。
- `github_repo/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py`：连续 buffer、bucket group 和分片通信状态。
- `github_repo/torchtitan/torchtitan/distributed/parallel_dims.py`：并行度与 device mesh 的一致性检查。

## 4. 常见错误判断

- 只看总通信字节，没有看 collective 次数和启动开销。
- 只在 world_size=1 跑通，就判断多 rank 梯度同步正确。
- 把 CPU/gloo 语义测试结果当成 NCCL 性能结论。
- 只调 bucket size，漏看 recompute、pipeline bubble、optimizer state 和 checkpoint I/O。
- 改 world size 后直接恢复分片 checkpoint，没有验证 optimizer state layout。

## 5. 结束条件

- 问题能被一个最小命令复现。
- bucket、grad diff、world size、并行度和关键指标已经落盘。
- 源码主路径中能指出状态在哪里产生、改变和通信。
- 结论写进 `training_step_template.md`，并包含下一步动作。
