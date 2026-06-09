# L10 源码阅读卡片

## 主路径

```text
patch ring_attention_forward
  -> tests compare with SDPA
  -> MiniInfra ring_attention_plan
  -> Megatron ModelParallelConfig
  -> parallel_state CP group
  -> Transformer Engine DotProductAttention CP kwargs
  -> run_seqlen / eval_yarn drill
```

## 必读片段

| 文件 | 行号 | 结论 |
|---|---:|---|
| `labs/l09_long_context_cp/patch/starter/ring_attention.py` | L34-L41 | starter 读 shape、scale，并要求沿序列维切 K/V |
| `labs/l09_long_context_cp/patch/starter/ring_attention.py` | L49-L60 | TODO 列出 online softmax 的 8 步更新 |
| `labs/l09_long_context_cp/patch/reference/ring_attention.py` | L29-L47 | reference 展示 scores、new max、old rescale、denom/out 和 normalize |
| `labs/l09_long_context_cp/patch/tests/test_patch.py` | L53-L75 | 测试覆盖 uneven chunk 和长 K/V |
| `mini_infra/megatron/core/context_parallel/ring_attention.py` | L59-L94 | 教学版 ring plan 计算 chunk、bytes、steps 和 CP 后 peak memory |
| `github_repo/Megatron-LM/megatron/core/model_parallel_config.py` | L45-L63 | Megatron 配置 context parallel size、hierarchical CP 和每 rank 最大序列长度 |
| `github_repo/Megatron-LM/megatron/core/parallel_state.py` | L953-L980 | Megatron 创建并保存 CP group 与 global ranks |
| `github_repo/Megatron-LM/megatron/core/extensions/transformer_engine.py` | L1437-L1464 | TE attention 在 CP 开启时接收 cp_group、global ranks、stream 和 comm type |
| `mini_infra/megatron/core/context_parallel/rope.py` | L27-L47 | RoPE 的反频率和二维旋转 |
| `mini_infra/megatron/core/context_parallel/yarn.py` | L29-L55 | YaRN 简化 scale、temperature 和 summary |

## 读源码时的判断句

- `num_chunks` 只改变 K/V 的读取顺序，正确实现下输出语义不变。
- 分块后不能对每个 chunk 独立 softmax 再相加，必须合并到同一个 denominator。
- 旧的 `running_denom` 和 `running_out` 必须同时乘 `exp(old_max - new_max)`。
- Context Parallel 的通信搬运 K/V chunk；online softmax 负责数学等价。
- RoPE/YaRN 处理位置外推证据，不能替代 attention 激活显存分析。

## 自检问题

1. `scores_chunk` 的 shape 是什么？
2. 第一次循环时 `running_max=-inf` 怎么安全处理？
3. `Sk=7, num_chunks=2` 时每个 chunk 可能多长？
4. Megatron 从哪里得到 CP group 和 global ranks？
5. `eval_yarn.py` 的输出为什么不能直接证明长上下文质量？
