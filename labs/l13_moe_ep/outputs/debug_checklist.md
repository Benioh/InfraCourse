# L14 Debug Checklist：MoE Router、Capacity 与 EP 通信

## 1. 先固定现场

- 记录命令、配置文件、git commit、Python/PyTorch 版本、硬件、world size、expert 数、top-k、capacity factor、EP/TP/DP 组合和随机种子。
- 保存 stdout/stderr、routing 指标、tokens per expert、drop rate、all-to-all latency、loss、checkpoint 和 report。
- 明确这是 patch-test、MiniInfra drill、notebook、EP smoke，还是真实 Megatron 训练。

## 2. 判断问题在哪一层

| 层 | 要看什么 | 可能结论 |
|---|---|---|
| 输入 | logits shape、num experts、top-k、capacity factor | 路由输入或容量配置已经不合理 |
| Router | entropy、aux loss、top-k 分布、tokens per expert | router collapse 或 expert load 不均 |
| Capacity | drop rate、overflow expert、combine weights 行和 | 容量过严或 drop 逻辑错位 |
| Dispatcher | EP group、all-to-all latency、token permutation、combine | 通信或 token 重排成为瓶颈 |
| 输出 | loss、grad、step time、expert utilization | 结果无法支撑当前判断 |

## 3. 沿源码主路径复查

- `labs/l13_moe_ep/patch/starter/moe_router.py`：学生需要补齐的 top-2 router 合同。
- `labs/l13_moe_ep/patch/reference/moe_router.py`：参考实现中的 softmax、top-k、capacity 和 aux loss。
- `labs/l13_moe_ep/patch/tests/test_patch.py`：张量不变量和边界 case。
- `mini_infra/megatron/core/transformer/moe/router.py`：教学版 routing、entropy 和 aux 指标。
- `mini_infra/megatron/core/transformer/moe/experts.py`：MoE summary 的指标聚合。
- `github_repo/Megatron-LM/megatron/core/transformer/moe/router.py`：Megatron `TopKRouter` 主路径。
- `github_repo/Megatron-LM/megatron/core/transformer/moe/moe_utils.py`：scatter、capacity dropping 和 aux loss。
- `github_repo/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py`：dispatcher 和通信边界。

## 4. 常见错误判断

- 只看 aux loss，不看 tokens per expert 和 drop rate。
- capacity drop 时只清 mask，忘记清 combine weights。
- 把 top-2 均匀 aux loss 误判成应该接近 1。
- 把 CPU patch 通过写成 EP all-to-all 性能合格。
- 只调 capacity factor，漏看 logits dtype、router collapse 和 EP group 配置。

## 5. 结束条件

- 问题能用一个最小命令或 profile 复现。
- routing 指标、capacity、drop rate、all-to-all 和 loss 证据已经落盘。
- 源码主路径中能指出状态在哪里生成、过滤、通信和合并。
- 结论写进 `moe_routing_template.md`，并包含下一步动作。
