# L14：MoE Top-2 Router、Capacity 与 Expert Parallelism

## 本讲目标

- 理解 MoE 为什么把 dense FFN 的计算问题变成路由、容量和通信问题。
- 能解释 router logits、softmax、top-2、dispatch mask 和 combine weights 的张量关系。
- 能按 `capacity_factor * num_tokens * top_k / num_experts` 推导 expert capacity，并说明 token dropping 的代价。
- 能实现 Switch 风格 aux loss，并解释 top-2 情况下均匀值为什么接近 2。
- 能把 patch reference 映射到 Megatron `TopKRouter`、token dropping 和 token dispatcher。

## 1. 问题背景：稀疏激活带来新的控制面

Dense Transformer 的 FFN 层让每个 token 经过同一组参数。MoE 把 FFN 换成多个 expert，每个 token 只选择少数 expert。这样做的收益是稀疏激活：模型可以拥有更多 expert 参数，但单个 token 只触发 top-k 个 expert 的计算。

这个收益不是免费的。router 可能把大量 token 分给同一个 expert，导致这个 expert 所在 rank 变成慢点；其他 expert 空闲，参数容量没有被使用。在 Expert Parallelism（EP）中，token 还要通过 all-to-all 发到 expert 所在 rank，计算完成后再发回原 rank。路由不均会同时影响模型质量、drop rate、通信量和 step time。

本关 patch 只实现 top-2 router 的张量合同。它不实现 expert MLP，也不执行真实 all-to-all。这样设计是为了先把 token-to-expert 对齐关系、capacity 和 aux loss 写对，再读生产框架的 dispatcher 分支。

**小检查：**

1. MoE 省下的是哪部分 dense FFN 计算？
2. 为什么 router 不均会让 EP group 的 step time 变慢？
3. 本关 patch 为什么不需要 GPU 也能验证核心语义？

## 2. Top-2 Router 的输入和输出

router logits 是形状为 `(num_tokens, num_experts)` 的打分矩阵。每一行对应一个 token，每一列对应一个 expert。对最后一维做 softmax 后，得到每个 token 在所有 expert 上的概率分布。`torch.topk(probs, k=2, dim=-1)` 选出两个 expert id 和对应概率。

top-2 会产生两个用途不同的结果。`dispatch_mask` 是 0/1 矩阵，表示 token 是否发给某个 expert；`combine_weights` 是同形状的浮点矩阵，表示 expert 输出回来后按多少权重合并。没有 capacity drop 时，每个 token 的 `dispatch_mask` 行和是 2，`combine_weights` 行和接近 1。

top-2 内部归一化很关键。softmax 给的是全 expert 分布，选出两个 expert 后，需要用两个选中概率的和重新归一化。否则只保留 top-2 后，combine weights 的行和会小于 1，MoE 分支输出尺度会随着未选中 expert 的概率质量漂移。

**小检查：**

1. `logits`、`top2_idx`、`dispatch_mask` 的形状分别是什么？
2. 为什么 `dispatch_mask` 和 `combine_weights` 要保持同一形状？
3. top-2 权重为什么要重新归一化？

## 3. Capacity Factor：每个 expert 的队列上限

capacity factor 控制每个 expert 在一个 batch 内最多接收多少 token route。top-2 时，路由槽位总数是 `num_tokens * 2`，均匀分给 `num_experts` 后，再乘以 `capacity_factor`。本关使用：

```text
capacity = max(1, int(capacity_factor * num_tokens * 2 / num_experts))
```

例如 32 个 token、4 个 expert、top-2、capacity factor 为 0.5 时，总路由槽位是 64，均匀每个 expert 是 16，乘以 0.5 后 capacity 是 8。任何 expert 最多保留 8 条 token route。超出 capacity 的 route 会被 drop。

drop 策略也有工程含义。本关按该 expert 列里的 combine weight 排序，保留权重最高的 token route，把其他 route 的 `dispatch_mask` 和 `combine_weights` 清零。这样可以保留 router 最有信心的分配，同时避免重新路由带来的额外通信轮。代价是被 drop 的 MoE 分支没有 expert 输出；如果 token 的两个 route 都被 drop，残差连接会让它近似跳过这一层 MoE。

capacity 太小会提高 drop rate，损害样本利用；capacity 太大又会扩大 expert buffer、padding 和通信量。训练通常会在 drop rate、质量和吞吐之间选一个折中点。

**小检查：**

1. 128 个 token、8 个 expert、top-2、capacity factor 为 1 时 capacity 是多少？
2. capacity drop 时为什么要同步清零 mask 和 weights？
3. capacity 太大和太小分别会带来什么代价？

## 4. Aux Loss：让 router 看到负载不均

Switch 风格 aux loss 用两个分布相乘：

```text
fraction_routed[i] = mean(dispatch_mask[:, i])
fraction_prob[i]   = mean(softmax(logits)[:, i])
aux_loss = num_experts * sum(fraction_routed * fraction_prob)
```

`fraction_routed` 来自离散路由结果，表示某个 expert 实际接到了多少 route；`fraction_prob` 来自 softmax 概率，保留可微梯度通道。两者相乘后，如果某个 expert 既被大量选中，又拿到高平均概率，aux loss 会放大，训练会推动 router 把概率分散到其他 expert。

top-k 时要注意尺度。top-2 下，均匀时每个 expert 的 `fraction_routed` 大约是 `2 / num_experts`，`fraction_prob` 大约是 `1 / num_experts`，乘积求和再乘 `num_experts` 后接近 2。本关沿用这个尺度，不额外除以 top-k。测试因此要求 near-uniform logits 的 aux loss 小于 2.5，极端偏向 expert 0 时明显大于 2。

aux loss 不是唯一指标。实际排查 router collapse 时，还要同时看 tokens per expert、capacity overflow rate、router entropy 和每个 expert rank 的 step time。如果 aux loss 正常但某些 expert 长期为空，可能是 capacity、routing dtype、batch 形状或 dispatcher 分桶出了问题。

**小检查：**

1. `fraction_routed` 和 `fraction_prob` 分别来自哪个张量？
2. top-2 均匀路由时 aux loss 为什么接近 2？
3. router collapse 时你会同时看哪三个指标？

## 5. 从 Patch 到 Megatron 的同构关系

patch reference 的主线很短：softmax 得到 `probs`，top-k 得到 `top2_vals/top2_idx`，对 top-2 weights 归一化，scatter 回完整 expert 维度，得到 `combine_weights` 和 `dispatch_mask`，再应用 capacity，最后计算 aux loss。

Megatron 的生产路径用了不同命名，但结构一致。`TopKRouter` 先通过 gating linear 生成 logits，再根据 score function 得到 scores 和 top-k routing map。`moe_utils.py` 会把 top-k 结果 scatter 回 dense `routing_probs` 和 boolean `routing_map`，再在 `apply_router_token_dropping` 中按 capacity factor 生成 capacity mask。

生产系统多出的复杂度主要在 dispatcher。`MoETokenDispatcher` 把 router 产物分成 dispatch preprocess、token dispatch、dispatch postprocess、combine preprocess、token combine、combine postprocess 等阶段。AllGather 或 AllToAll 分支会把 token 发到 expert 所在 rank，并把输出按原 token 顺序组合回来。本关 patch 不覆盖这些通信，只提供 dispatcher 需要的正确路由张量。

**小检查：**

1. patch 的 `dispatch_mask` 对应 Megatron 哪个张量？
2. patch 的 `combine_weights` 对应 Megatron 哪个张量？
3. dispatcher 为什么需要同时知道 routing map 和 routing probabilities？

## 6. 工程验证：patch-test 与 MiniInfra drill

`make patch-test M=l13_moe_ep` 验证五个最小合同：每个 token 最多两个 expert；没有 capacity drop 时 weights 行和接近 1；capacity 小时 expert load 不超过上限；均匀 logits 的 aux loss 接近 top-k；极端不均时 aux loss 变大。这些测试全部在 CPU 上跑，证明张量语义，不证明 EP 性能。

MiniInfra drill 用 `run_moe.py` 输出一组 routing 指标：router entropy、aux loss、capacity overflow rate、tokens per expert 分位数和 all-to-all 时间估算。它适合练习如何解释路由现象。例如 router entropy 低、p99 tokens per expert 高、capacity overflow rate 高，通常说明 token 分布集中在少数 expert。

真实训练还要补充 GPU 侧证据：all-to-all latency、expert MLP 时间、drop rate、loss 曲线和每个 rank 的负载。讨论“MoE 更快”时必须写明比较对象、模型规模、top-k、expert 数、EP/TP/DP 组合、batch 和网络条件。

**小检查：**

1. patch-test 和 MiniInfra drill 分别证明什么？
2. 哪些指标能提示 router collapse？
3. 对比 MoE 与 dense FFN 时必须记录哪些条件？

## 7. 课后思考

1. 如果 top-2 的两个 route 都被 capacity drop，这个 token 的 MoE 分支输出会怎样？
2. 如果 all-to-all 很慢，但 tokens per expert 很均匀，下一步应检查 dispatcher、网络还是 aux loss？
3. 如果把 top-k 从 2 改成 1，capacity 公式、aux loss 均匀值和通信量会怎样变化？

## 8. 小结

- MoE 用稀疏激活减少每个 token 的 expert 计算，但引入 router、capacity 和 EP 通信边界。
- `dispatch_mask` 决定 token 发给谁，`combine_weights` 决定 expert 输出回来后怎么合并，两者必须共享同一 token-to-expert 对齐关系。
- capacity factor 是 expert 队列上限，太小会 drop token，太大增加 buffer、padding 和通信成本。
- Switch 风格 aux loss 同时使用真实 route fraction 和可微 softmax probability；top-2 均匀值接近 2。
- 本关 patch 只验证 router 张量合同；真实 MoE 性能还要看 dispatcher、all-to-all、expert load 和训练日志。
