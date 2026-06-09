# L15：Pipeline Parallel 1F1B 调度

## 0. 本讲目标

- 理解 Pipeline Parallelism 为什么需要 stage 和 microbatch。
- 能解释 GPipe、1F1B、interleaved 1F1B 的核心差异。
- 能手写 non-interleaved 1F1B 的 per-stage timeline。
- 能把 warmup、steady、cooldown 对应到 Megatron 的真实 schedule 代码。
- 能用 bubble ratio、stage idle time 和 activation 驻留窗口判断 PP 配置是否合理。

## 1. 问题入口：为什么要把一个训练 step 切成流水线

当模型层数和 hidden size 继续增大时，单卡或单个 FSDP shard 仍可能放不下完整前向和反向所需的参数、激活和中间 tensor。Pipeline Parallelism 的做法是按层把模型切成多个 stage：stage 0 持有前几层，stage 1 持有后续层，直到最后一个 stage 输出 loss。

如果只把一个 batch 直接送进 pipeline，会出现大量设备等待：stage 1 要等 stage 0 产出 activation，stage 2 要等 stage 1，反向时又反过来等待梯度。microbatch 的作用是把 global batch 切成多个小批次，让 stage 0 在 stage 1 处理 microbatch 0 时继续处理 microbatch 1，从而让多个 stage 同时工作。

PP 的收益不是减少总 FLOPs。它改善的是跨设备利用率和单 stage 显存压力。代价是 P2P activation / gradient 通信、pipeline bubble、microbatch 对优化器统计的影响，以及调度实现复杂度。

## 2. 三个基本对象：stage、microbatch、timeline

**Pipeline stage** 是模型层切分后的连续片段。每个 stage 只负责自己的层，并和前后 stage 交换 activation 或 gradient。

**Microbatch** 是 global batch 的切片。训练语义上，一个 optimizer step 仍对应多个 microbatch 的梯度累积；调度语义上，microbatch 是 pipeline 里流动的工作单元。

**Timeline** 是某个 stage 本地执行的 op 序列。L15 patch 里用 `("F", i)` 表示 microbatch `i` 的 forward，用 `("B", i)` 表示对应 backward。

这三个对象连起来后，1F1B 的输入是 `num_stages = D` 和 `num_microbatches = N`；中间状态是每个 stage 已经 forward 或 backward 到哪个 microbatch；输出是 `schedule[stage] = [(op, microbatch), ...]`。

## 3. GPipe 的问题：激活驻留太长

GPipe 风格的朴素调度先做所有 microbatch 的 forward，再做所有 backward。它容易解释，但内存压力大：某个 stage 做完 microbatch 0 的 forward 后，必须等许多后续 forward 完成，才能收到 microbatch 0 的反向梯度。这个等待期间，microbatch 0 的激活要一直留着。

如果 `N` 很大，GPipe 的激活驻留窗口接近 `N` 个 microbatch。它可以把 pipeline 填满，但把显存压力推高。1F1B 的动机就是在 pipeline 填起来后尽早交替 backward，让已经具备反向依赖的 microbatch 及时释放激活。

## 4. 1F1B 三段式：warmup、steady、cooldown

non-interleaved 1F1B 分三段。

**Warmup**：前面的 stage 先连续做 forward，把 activation 送往后面的 stage。stage `s` 的 warmup forward 数是 `D - s - 1`。stage 越靠后，越晚接到 activation，warmup 越短。

**Steady**：进入稳定阶段后，每个 stage 按 `F, B, F, B, ...` 交替。一次 forward 继续把新 microbatch 往后送；一次 backward 处理已经从后面回来的梯度。这个阶段是 1F1B 名字的来源。

**Cooldown**：所有 forward 都排入 pipeline 后，还剩下一批 backward 要从最后 stage 反向排空。stage `s` 的 cooldown backward 数和 warmup 对称，也是 `D - s - 1`。

对单个 stage 来说，timeline 长度应为 `2 * N`：每个 microbatch 有一次 forward 和一次 backward。patch 不模拟跨 stage 的真实时间戳，但它要求本地顺序成立：forward index 递增，backward index 递增，且同一个 microbatch 的 backward 不能早于 forward。

## 5. Bubble：首尾空泡为什么是 `2 * (D - 1)`

Pipeline bubble 指设备因为依赖未满足而空闲的阶段时间。标准 non-interleaved 1F1B 有两段固定空泡：开头要花 `D - 1` 个 stage time 填满 pipeline，结尾要花 `D - 1` 个 stage time 排空反向。因此 patch 中的 `bubble_count(D)` 返回 `2 * (D - 1)`。

Bubble ratio 会随 microbatch 数增加而下降，因为固定空泡被更多有效工作摊薄。比较时要说明条件：在同样 stage 数、forward/backward 耗时模型和 global batch 语义下，增加 `N` 通常降低 bubble ratio；但它会改变 microbatch size、activation 窗口、通信次数和梯度累积粒度。真实训练里不能只用 bubble ratio 选配置，还要看 loss 稳定性、P2P 等待、显存峰值和 optimizer step time。

## 6. Patch 合同：生成局部 timeline

本关实现两个函数：

```python
def make_1f1b_schedule(num_stages: int, num_microbatches: int) -> list[list[tuple[str, int]]]:
    ...

def bubble_count(num_stages: int) -> int:
    ...
```

`num_stages` 必须为正，`num_microbatches` 必须大于等于 `num_stages`。这个约束来自填充 pipeline 的基本需求：如果 microbatch 数少于 stage 数，后面 stage 很难进入稳定 1F1B 区间，测试要求显式抛 `ValueError`。

参考实现对每个 stage 分三段写 timeline：

1. 先追加 `warmup` 个 forward。
2. 再运行 `N - warmup` 次 steady，每次追加一个 forward 和一个 backward。
3. 最后追加 `warmup` 个 backward。

这段实现没有 P2P 通信、没有 loss、没有 activation tensor。它只检查调度顺序。如果这个最小合同写错，真实 Megatron schedule 中的 send/recv、activation 保存和 backward 依赖都会更难排查。

## 7. MiniInfra 到 Megatron：同一条主线，不同复杂度

MiniInfra 的 `forward_backward_pipelining_without_interleaving` 用事件列表表示 forward 和 backward。它把 microbatch 和 stage 组合成 `PipelineEvent`，适合建立直觉：forward 从 stage 0 往后推进，backward 从最后 stage 往前回流。

Megatron 的真实 schedule 更复杂。`get_forward_backward_func` 先根据 PP size 和 virtual PP size 选择 no pipeline、non-interleaved pipeline 或 interleaved pipeline。进入 `forward_backward_pipelining_without_interleaving` 后，代码会创建 P2P communicator，计算 warmup microbatch 数，接收 forward tensor，执行 forward step，发送 activation，再接收 backward gradient 并执行 backward step。稳定阶段里，`send_forward_recv_backward` 和 `send_backward_recv_forward` 把通信与调度依赖绑在一起。

生产代码还要处理 activation pseudo-deallocate、gradient sync、embedding gradient、context parallel、multi-module pipeline、activation checkpoint 和 cuda graph。L15 不要求一次读完这些分支，但要知道它们围绕同一条主线服务：让每个 stage 在正确时间拿到 input activation 或 output gradient。

## 8. Drill：怎样解释一次 PP smoke

`scripts/run_pp_smoke.py` 用 mock forward/backward time 计算 schedule 的 `longest_stage_seconds`、`bubble_ratio`、`total_ops` 和 `accept`。默认 CPU 配置是 4 stages、8 microbatches，理论 bubble count 是 6，总 op 数是 `4 * 8 * 2 = 64`。

看结果时不要只看 accept。要同时记录：

- `num_stages` 和 `num_microbatches`：决定理论 bubble 和总 op 数。
- `forward_time` 和 `backward_time`：决定 mock makespan。
- `bubble_ratio`：判断空泡占比是否符合预期。
- `schedule_head` 和 `schedule_last`：检查前后 stage 的 warmup/cooldown 形态。
- Megatron 命令模板：确认真实训练时 PP size、microbatch 和 virtual PP 配置一致。

如果 bubble ratio 高，先检查 microbatch 数是否太少、warmup 是否 off-by-one、steady 阶段是否出现连续 forward 或连续 backward，再看真实 P2P wait 是否把理论 schedule 拉长。

## 9. 小结

L15 的知识链路是：模型按层切成 stage，global batch 切成 microbatch，warmup 负责填充 pipeline，steady 阶段用 1F1B 缩短激活驻留，cooldown 排空剩余 backward，bubble 是首尾依赖空闲的代价。patch 验证的是 timeline 不变量；Megatron 源码把同一套顺序扩展成真实 tensor 通信、activation 管理和梯度收尾。
