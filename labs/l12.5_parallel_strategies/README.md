# L12.5 · 并行策略全景对比

这一讲在学完所有单独的并行方式（DDP L03、TP L05、AC L06、FSDP L12、MoE/EP L13、PP L14）之后，做一次系统的全景对比。目标是让你面对一个真实的训练任务时，能回答"我应该用哪种并行组合"这个问题。

前面的 Lab 每个只教一种并行方式。但真实工程中你面对的问题是："7B 模型单机 8 卡能装下吗？""70B 模型需要 TP 还是 FSDP？""跨机通信慢，应该用 PP 还是 DP？" 这些问题需要把所有并行方式放在一个框架里对比。

## 学习路线

1. 读 [lecture.md](lecture.md)：并行策略全景图 + 决策框架。
2. 做 quiz：确认你能选择正确的并行策略组合。
3. 做 patch：实现并行策略分析和决策函数。
4. 填写 [outputs/parallel_strategy_report.md](outputs/parallel_strategy_report.md)。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Distributed training strategy |
| 它解决什么问题 | 面对具体硬件和模型规模，选择最优并行组合 |
| lab 检验什么 | 显存估算、通信量计算、策略选择 |

## 你会学到什么

### 五种集合通信原语对比
- **all_reduce**：所有 rank 得到相同的聚合结果（如梯度求和）。通信量 2×N（ring）。
- **all_gather**：每个 rank 有一份，结果是所有 rank 的拼接。通信量 N×(ws-1)/ws。
- **reduce_scatter**：先 reduce 再 scatter，每个 rank 得到结果的一部分。通信量 N×(ws-1)/ws。
- **all_to_all**：每个 rank 发不同数据给不同 rank。通信量取决于 routing。
- **broadcast**：一个 rank 的数据广播给所有 rank。通信量 N。

### 并行策略：省了什么、代价是什么

| 策略 | 省了什么 | 通信代价 | 通信能否 overlap | 适合场景 |
|---|---|---|---|---|
| **DDP** | 无（数据并行） | all-reduce gradients | 可以（bucket backward） | 模型能放单卡 |
| **ZeRO-1** | optimizer state | all-reduce gradients + broadcast opt state | 部分 | 模型能放单卡，optimizer state 占太多 |
| **ZeRO-2** | optimizer + gradients | reduce-scatter grad + all-gather opt state | 部分 | 模型能放单卡 |
| **ZeRO-3 / FSDP** | optimizer + gradients + parameters | all-gather params (fwd+bwd) + reduce-scatter grad | 可以（prefetch） | 模型不能放单卡 |
| **TP** | 模型参数（列/行切分） | all-reduce / all-gather per layer | 困难（在计算路径上） | 单层参数太大 |
| **PP** | 模型层（按 stage 切） | 点对点通信 activation | 有 bubble | 模型层数多、跨机带宽低 |
| **SP** | activation（序列维度切） | all-gather / reduce-scatter activation | 和 TP 配合 | 长序列 + TP |
| **CP** | activation（ring 传 KV） | ring send/recv KV chunks | 和计算 overlap | 超长序列 attention |
| **EP** | MoE 层参数 | all-to-all token routing | 取决于 capacity | MoE 模型 |

### 显存估算方法

一个模型的训练显存 = 参数 + 梯度 + 优化器状态 + 激活

```
参数: P bytes (FP16: P×2, FP32: P×4)
梯度: 和参数等大 (FP16/FP32)
Adam 优化器: 2×P (momentum + variance, FP32)
激活: 取决于 batch size × seq_len × hidden_size × num_layers
```

对于 FP16 + Adam：
- 参数 + 梯度 + optimizer = 2P + 2P + 2×4P = 12P bytes（混合精度：2+2+4+4=12 bytes/param）
- 激活 ≈ batch × seq × hidden × layers × factor

### 决策框架

```
1. 估算模型大小和显存需求
2. 判断单卡能否放下：
   - 能放下（参数 + optimizer ≤ 单卡显存的 70%）→ DDP / ZeRO-1
   - 放不下 → 需要模型并行
3. 选择模型并行方式：
   - 单机内（NVLink 带宽高）→ TP + FSDP
   - 跨机（网络带宽低）→ PP + DP
   - 超长序列 → + SP 或 CP
   - MoE → + EP
4. 验证通信不是瓶颈（profiler）
5. 验证 bubble 可接受（PP 场景）
```

## Patch 闭环

```bash
make patch-test M=l12.5_parallel_strategies
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_memory_estimate` | 能正确估算模型训练显存 |
| `test_comm_volume_allreduce` | all-reduce 通信量计算正确 |
| `test_comm_volume_reduce_scatter` | reduce-scatter 通信量计算正确 |
| `test_strategy_selection_small` | 小模型正确选择 DDP |
| `test_strategy_selection_large` | 大模型正确选择 TP+FSDP |
| `test_zero_memory_saving` | ZeRO-1/2/3 各自的显存节省计算正确 |

## 进入下一讲

通过后进入 [L13 MoE/EP](../l13_moe_ep/README.md)。
