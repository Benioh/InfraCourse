# L14 Source Reading Card：MoE Router 与 EP

## 主路径

1. `labs/l13_moe_ep/patch/starter/moe_router.py`：学生需要补齐的 top-2 router 合同。
2. `labs/l13_moe_ep/patch/reference/moe_router.py`：softmax、top-k、scatter、capacity、aux loss。
3. `labs/l13_moe_ep/patch/tests/test_patch.py`：mask、weights、capacity 和 aux loss 验收。
4. `mini_infra/megatron/core/transformer/moe/router.py`：教学版 routing、entropy 和 aux 指标。
5. `mini_infra/megatron/core/transformer/moe/experts.py`：MoE 指标聚合。
6. `github_repo/Megatron-LM/megatron/core/transformer/moe/router.py`：Megatron `Router` 和 `TopKRouter`。
7. `github_repo/Megatron-LM/megatron/core/transformer/moe/moe_utils.py`：routing scatter、capacity dropping、aux loss。
8. `github_repo/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py`：dispatch 和 combine 通信边界。

## 阅读方法

1. 先看 patch 签名，确认输入是 logits 和 capacity factor。
2. 再看 reference 的 softmax、top-k、scatter 和 capacity drop。
3. 用 tests 确认本关验收的是哪些张量不变量。
4. 用 MiniInfra drill 把路由结果转成 entropy、drop rate 和 all-to-all 估算。
5. 最后读 Megatron，确认生产路径里多出的 process group、dispatcher、padding 和 overlap。

## 自检

- 我能否解释 `dispatch_mask`、`combine_weights`、`routing_map` 和 `routing_probs` 的对应关系？
- 我能否算出给定 top-k 和 capacity factor 下的 expert capacity？
- 我能否说明 top-2 均匀 aux loss 为什么接近 2？
- 我能否区分 router 问题、capacity 问题和 dispatcher 通信问题？
