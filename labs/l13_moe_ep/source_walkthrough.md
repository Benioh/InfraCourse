# L14 源码带读：MoE Top-2 Router 与 Dispatcher

这份带读按“patch 最小合同 → MiniInfra 指标 → Megatron 生产路径”的顺序走。先读 [lecture.md](lecture.md)，再按下面的锚点看源码。

## 0. 源码地图

```text
labs/l13_moe_ep/patch/starter/moe_router.py
  -> labs/l13_moe_ep/patch/reference/moe_router.py
  -> labs/l13_moe_ep/patch/tests/test_patch.py

mini_infra/megatron/core/transformer/moe/router.py
  -> mini_infra/megatron/core/transformer/moe/experts.py
  -> labs/l13_moe_ep/scripts/run_moe.py

github_repo/Megatron-LM/megatron/core/transformer/moe/router.py
  -> github_repo/Megatron-LM/megatron/core/transformer/moe/moe_utils.py
  -> github_repo/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py
```

patch 负责缩小 top-2 routing 的张量合同；MiniInfra 负责生成路由指标；Megatron 展示生产系统如何把 routing map 交给 dispatcher。

## 1. Patch starter：看接口和 TODO

文件：`labs/l13_moe_ep/patch/starter/moe_router.py`

重点看：

- L20-L30: `top2_router` 的输入输出合同。
- L33-L46: TODO 要求 softmax、top-2、归一化和 scatter。
- L48-L64: TODO 描述 capacity 限制和 overflow drop。
- L66-L73: TODO 描述 Switch 风格 aux loss。

读完后要得到的结论：本关实现的是 router 张量语义，不涉及 expert MLP 或通信。

可以先跳过：真实框架里的 jitter、bias、sinkhorn、fused router 和 all-to-all overlap。

## 2. Patch reference：看张量状态如何推进

文件：`labs/l13_moe_ep/patch/reference/moe_router.py`

重点看：

- L16-L22: softmax、top-2、top-2 内归一化、scatter 和 mask。
- L24-L33: capacity 计算和每个 expert 的 overflow drop。
- L35-L39: `fraction_routed`、`fraction_prob` 和 aux loss。

读完后要得到的结论：mask 和 weights 必须同步变化，aux loss 的概率来源是全 softmax。

## 3. Patch tests：确认验收边界

文件：`labs/l13_moe_ep/patch/tests/test_patch.py`

重点看：

- L22-L30: capacity 充足时每个 token 应有两个 active expert。
- L33-L42: 没有 drop 时 combine weights 行和接近 1。
- L45-L62: capacity 小时，每个 expert load 不超过 capacity。
- L65-L72: near-uniform logits 的 aux loss 应接近 top-k。
- L75-L86: 极端偏向时 aux loss 应明显升高。

读完后要得到的结论：测试覆盖张量不变量，不覆盖 GPU all-to-all 性能。

## 4. MiniInfra router：看指标从哪里来

文件：`mini_infra/megatron/core/transformer/moe/router.py`

重点看：

- L28-L34: 教学版 softmax。
- L37-L50: `route_tokens` 对每个 token 做 top-k 选择。
- L53-L68: `tokens_per_expert` 和 `router_entropy` 反映 expert load 分布。
- L71-L82: 教学版 aux loss 用 token count 方差刻画不均。

读完后要得到的结论：MiniInfra 指标用于观察路由现象，不等于 Megatron 的完整 loss 公式。

## 5. MiniInfra summary 和 drill

文件：`mini_infra/megatron/core/transformer/moe/experts.py`

重点看：

- L30-L41: `moe_summary` 串起 synthetic logits、routing、capacity 和 dispatch plan。
- L44-L60: 返回 router entropy、aux loss、overflow、tokens per expert 和 all-to-all 估算。

文件：`labs/l13_moe_ep/scripts/run_moe.py`

重点看：

- L12-L18: CLI 接收 expert 数、top-k、EP size 和 TP size。
- L18-L20: 调用 `moe_summary` 并输出 JSON。

读完后要得到的结论：drill 产物是指标解释工具，不是真实性能 benchmark。

## 6. Megatron router：看生产命名

文件：`github_repo/Megatron-LM/megatron/core/transformer/moe/router.py`

重点看：

- L29-L37: `Router` 基类接收配置和 MoE process groups。
- L85-L107: `gating` 把 hidden states 转成 router logits。
- L137-L150: `TopKRouter` docstring 说明 logits、scores、probs 和 routing_map 的命名关系。
- L166-L170: `TopKRouter` 读取 top-k、load balancing type 和 score function。

读完后要得到的结论：Megatron 的 `routing_map` 对应 dispatch mask，`probs` 对应 combine 权重来源。

## 7. Megatron moe_utils：看 scatter、capacity 和 aux loss

文件：`github_repo/Megatron-LM/megatron/core/transformer/moe/moe_utils.py`

重点看：

- L59-L68: Switch load-balancing loss 的函数签名。
- L142-L146: aux loss 使用 expert 概率和 token count。
- L823-L842: top-k 结果 scatter 成 dense routing probabilities 和 boolean routing map。
- L904-L911: token dropping 的输入包括 routing probs、routing map、top-k 和 capacity factor。
- L933-L940: capacity 根据 `num_tokens * router_topk` 计算。
- L947-L965: 按概率或位置生成 capacity mask，并同步作用到 probs 和 map。

读完后要得到的结论：生产代码把 patch 的 dense weights/mask 拆成 routing_probs/routing_map，并在 capacity 后保持二者对齐。

## 8. Megatron token dispatcher：看通信边界

文件：`github_repo/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py`

重点看：

- L53-L81: dispatcher 保存 config、shared expert、EP/TP process group 和 rank 信息。
- L87-L108: `dispatch_preprocess` 只做本地准备，避免提前通信。
- L111-L125: `token_dispatch` 执行 token 到 expert 设备的通信。
- L149-L181: combine 阶段把 expert 输出跨设备合回。
- L211-L238: AllGather dispatcher 保存 local expert 和 top-k 配置。

读完后要得到的结论：router 决定 token-to-expert 关系，dispatcher 负责把这个关系变成通信和重排。

## 读完后的自检问题

1. patch reference 的哪几行保证 mask 和 weights 同步 drop？
2. MiniInfra 的 aux loss 和 Megatron 的 aux loss 有什么差异？
3. Megatron `routing_map` 和 `routing_probs` 分别对应 patch 的哪个输出？
4. dispatcher 的哪些步骤会影响 all-to-all latency？
5. 如果某个 expert 过载，你会先看 logits、capacity，还是 dispatcher 进程组？
