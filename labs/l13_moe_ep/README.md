# L14 · MoE / EP：Top-2 Router、Capacity 与 Aux Loss

这一讲看 Mixture of Experts（MoE）层里最容易出错的控制面：router 怎样把 token 分给 expert，capacity 怎样限制热 expert 的队列长度，aux loss 怎样把负载不均反馈给 router。MoE 的稀疏激活能减少每个 token 经过的 expert 数，但它把 dense FFN 的计算问题换成了路由、容量、drop rate 和 Expert Parallelism（EP）通信问题。

本关 patch 只实现 `top2_router(logits, capacity_factor)`。它返回 `dispatch_mask`、`combine_weights` 和 `aux_loss`，不实现 expert MLP，也不执行 NCCL all-to-all。讲授重点是把路由张量语义讲清楚，再对照 Megatron 的 `TopKRouter`、token dropping 和 dispatcher。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L14 在训练并行主线中的位置。
2. 读 [lecture.md](lecture.md)：理解 MoE 稀疏激活、top-2 routing、capacity、aux loss 和 EP 通信边界。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra、Megatron router、token dropping 和 dispatcher 主路径读源码。
4. 跑 notebook：[n15_moe_router_capacity.ipynb](../../notebooks/n15_moe_router_capacity.ipynb)，观察容量和路由不均。
5. 做 quiz：确认 top-k、capacity factor、aux loss 和 EP/TP 的边界。
6. 做 patch：实现 `top2_router` 并通过 CPU 测试。
7. 跑 drill：用 MiniInfra 输出 router entropy、capacity overflow 和 all-to-all 估算。
8. 填写 [outputs/moe_routing_template.md](outputs/moe_routing_template.md)，沉淀一次 MoE 路由复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems / sparse MoE / Expert Parallelism |
| 它承接什么 | L13 的分片训练视角：哪些状态分片、哪些 token 需要通信 |
| 它解决什么问题 | 用 top-2 router、capacity 和 aux loss 控制 MoE token-to-expert 分配 |
| 它连接哪些指标 | tokens per expert、router entropy、aux loss、drop rate、all-to-all cost、step time |
| 它连接哪些源码 | patch `moe_router.py`、MiniInfra MoE、Megatron `TopKRouter`、`moe_utils.py`、`token_dispatcher.py` |
| lab 检验什么 | top-2 选择、combine weights、capacity drop、Switch 风格 aux loss |

## 你会学到什么

- 解释 MoE 为什么把 FFN 计算问题变成路由和通信问题。
- 说清 `logits -> softmax -> top-k -> routing map/probs -> capacity -> aux loss` 的张量链路。
- 区分 `dispatch_mask`、`combine_weights`、Megatron 的 `routing_map` 和 `routing_probs`。
- 判断 capacity factor 太小或太大时对 drop rate、padding、显存和 all-to-all 的影响。
- 用 MiniInfra 和 Megatron 源码定位 router collapse、expert overload 和 dispatcher 慢的问题。

## Patch 闭环

```bash
cat labs/l13_moe_ep/patch/task.md
$EDITOR labs/l13_moe_ep/patch/starter/moe_router.py
make patch-test M=l13_moe_ep
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_each_token_routes_to_two` | capacity 足够时每个 token 走两个 expert |
| `test_combine_weights_sum_to_one_no_capacity` | 无 drop 时 combine weights 行和接近 1 |
| `test_capacity_factor_drops_overflow` | 热 expert 不超过 capacity |
| `test_aux_loss_low_when_balanced` | 均匀路由下 top-2 aux loss 接近 2 |
| `test_aux_loss_high_when_imbalanced` | 极端偏向时 aux loss 明显升高 |

## Drill 闭环

```bash
python labs/l13_moe_ep/scripts/run_moe.py --experts 8 --topk 2 --ep 2
```

drill 会输出 `router_entropy`、`aux_loss`、`capacity_overflow_rate`、`tokens_per_expert_p50/p99` 和 `alltoall_ms` 估算。它用于练习指标解释，不替代真实 EP all-to-all benchmark。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 router collapse、capacity overflow、expert imbalance 和 all-to-all 慢 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、MiniInfra、Megatron router 和 dispatcher 主路径 |
| [outputs/moe_routing_template.md](outputs/moe_routing_template.md) | 记录一次 MoE routing / EP smoke / benchmark 的配置、指标和判断 |

## 进入下一讲

`make patch-test M=l13_moe_ep` 通过，并完成一次 MoE routing drill 复盘后，进入 [L15 Pipeline Parallel 1F1B](../l14_pipeline_1f1b/README.md)。
