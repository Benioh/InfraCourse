# L10 讲义：长上下文 Context Parallel 与 Online Softmax

这一讲讲长上下文 attention 的计算和系统边界。

当序列长度从 4K 提到 32K 或 128K 时，训练最先撞到的瓶颈经常来自 attention 的中间激活，而参数量只是显存账本的一部分。标准 attention 会构造 `scores = QK^T / sqrt(D)`，形状是 `(B,H,Sq,Sk)`。`Sq` 和 `Sk` 同时变长时，score 张量按二次方增长。

L10 的核心问题是：不显式保存完整 score 矩阵，怎样得到和 full attention 数学等价的输出？Patch 的答案是 online softmax：把 K/V 切成 chunk，逐块更新 running max、running denominator 和未归一化输出。真实 Context Parallel 会把这些 K/V chunk 放在不同 rank 上沿 ring 传递，本关先在单进程里把数学合同写对。

## 1. 本讲目标

学完这一讲，你应该能回答：

1. full attention 的 `scores` shape 和显存量级怎样计算。
2. Q/K/V 的 `(B,H,S,D)` 四个维度分别是什么。
3. 为什么每个 chunk 的 softmax 输出不能直接相加。
4. online softmax 维护哪些 running 状态，旧状态怎样重缩放。
5. `num_chunks=1`、`num_chunks=4` 和 uneven chunks 为什么应得到同一 attention 语义。
6. Context Parallel 中 Q stationary、K/V ring、CP group 和 `cp_comm_type` 的关系。
7. RoPE/YaRN 解决的位置外推问题与 ring attention 的显存问题有什么边界。

## 2. 真实问题：score 激活按 `Sq * Sk` 增长

标准 attention 的核心计算是：

```text
scores = Q @ K^T / sqrt(D)       # (B,H,Sq,Sk)
weights = softmax(scores)        # (B,H,Sq,Sk)
out = weights @ V                # (B,H,Sq,D)
```

假设 `B=1`、`H=32`、`Sq=Sk=8192`、fp16，光 `scores` 或 `weights` 的量级就是：

```text
1 * 32 * 8192 * 8192 * 2 bytes ~= 4 GB
```

训练时还要考虑多层、反向传播保存、dropout、mask、Q/K/V 本身、MLP 激活和 optimizer state。序列继续增大后，attention 激活会很快压过参数显存。

分块 attention 的思路是每次只构造：

```text
scores_chunk = Q @ K_chunk^T / sqrt(D)   # (B,H,Sq,chunk_len)
```

峰值 score 激活由 `chunk_len` 决定。总 QK 关系仍要算完，所以总 FLOPs 不会因为 chunk 增多而消失；收益来自更低的峰值中间张量和更好的内存访问方式。

## 3. Q/K/V shape 先要写准

本关使用 head-major 的张量格式：

```text
q: (B,H,Sq,D)
k: (B,H,Sk,D)
v: (B,H,Sk,D)
```

其中：

| 维度 | 含义 |
|---|---|
| `B` | batch size |
| `H` | attention heads |
| `Sq` | query 序列长度 |
| `Sk` | key/value 序列长度 |
| `D` | 每个 head 的维度 |

计算分数时：

```python
scores = torch.einsum("bhid,bhjd->bhij", q, k_chunk) * scale
```

`i` 是 query 位置，`j` 是当前 K chunk 的位置。输出始终沿 query 位置返回：

```text
out: (B,H,Sq,D)
```

`Sq` 和 `Sk` 可以不同。本关 `test_long_seq` 使用 `Sq=256`、`Sk=1024`，就是为了防止实现写死 square attention。

## 4. 为什么分块 softmax 不能直接相加

如果把 K/V 切成两块，错误做法是：

```text
out = softmax(QK_0^T) @ V_0 + softmax(QK_1^T) @ V_1
```

这个结果一般不等于 full attention。原因是 softmax 的分母应该覆盖所有 key：

```text
sum_j exp(score_j)
```

每个 chunk 单独 softmax 时，分母只覆盖当前 chunk。把它们直接相加，相当于让每块都有自己的归一化体系，最终权重总和也不会对齐 full attention。

正确做法是在所有 chunk 之间共享一个运行中的 softmax 状态。对每个 query 位置维护：

```text
running_max     # 当前看过的所有 key 分数的最大值
running_denom   # 在 running_max 参考下累加的 exp 分母
running_out     # 在 running_max 参考下累加的 exp(score) @ V
```

最终输出是：

```text
running_out / running_denom
```

`running_denom` 的 shape 是 `(B,H,Sq,1)`，除法时沿 `D` 维广播。

## 5. Online softmax 的等价更新

每读一个 chunk：

```text
scores = Q @ K_chunk^T / sqrt(D)
chunk_max = max(scores, dim=-1, keepdim=True)
new_max = max(running_max, chunk_max)
```

如果新 chunk 的最大值更大，旧状态过去使用的是旧 `running_max` 作为指数平移参考。为了把旧状态换到 `new_max` 参考下，需要缩放：

```text
exp_old = exp(running_max - new_max)
```

然后更新：

```text
exp_chunk = exp(scores - new_max)
running_denom = running_denom * exp_old + sum(exp_chunk)
running_out = running_out * exp_old + exp_chunk @ V_chunk
running_max = new_max
```

这里有两个关键点。

第一，`running_denom` 和 `running_out` 必须使用同一个 `exp_old` 缩放。只缩放分母或只缩放输出都会破坏等价性。

第二，第一次循环时 `running_max = -inf`。参考实现用 `torch.where(torch.isfinite(running_max), exp(...), 0)` 避免 `-inf - finite` 的边界产生不清晰的数值路径。由于旧分母和旧输出都初始化为 0，第一次旧状态贡献应该为 0。

## 6. K/V chunk 与 uneven chunk

本关用：

```python
k_chunks = list(torch.chunk(k, num_chunks, dim=2))
v_chunks = list(torch.chunk(v, num_chunks, dim=2))
```

`dim=2` 是序列维。`torch.chunk` 会处理不能整除的情况。例如 `Sk=7`、`num_chunks=2` 时，chunk 长度会是不均匀的。实现不能假设每块长度相同，只需要让 einsum 接受当前 `chunk_len`：

```python
scores: (B,H,Sq,chunk_len)
exp_chunk @ v_chunk -> (B,H,Sq,D)
```

`num_chunks` 改变的是计算顺序和峰值中间张量，不应该改变数学语义。测试会分别用 1 块、4 块、不均匀块和长 K/V 来抓这个边界。

## 7. 从本地 chunk 到 Context Parallel ring

Patch 里的循环在一个进程里完成：

```text
for k_chunk, v_chunk in local_chunks:
    update online softmax
```

Context Parallel 把 K/V chunk 分布到多个 rank。每个 rank 保持自己的 Q 分片不动，K/V chunk 沿 ring 传递：

```text
step 0: 每个 rank 处理自己的 K/V chunk
step 1: 每个 rank 接收左邻 rank 的 K/V chunk
step 2: 继续接收下一个 K/V chunk
...
```

如果 CP world size 是 `cp_size`，每个 rank 需要见过所有 K/V chunk。除去自己已经有的一块，ring 需要 `cp_size - 1` 次跨 rank 传递。MiniInfra 的 `ring_attention_plan` 会把这件事展开成可读的 `RingStep` 列表，并估算 `attn_comm_bytes`。

真实 Megatron/TE 的职责分工大致是：

| 组件 | 职责 |
|---|---|
| `ModelParallelConfig.context_parallel_size` | 指明序列维被切成多少个 CP rank |
| `parallel_state` | 创建和返回 CP process group |
| TE DotProductAttention | 接收 `cp_group`、`cp_global_ranks`、`cp_stream`、`cp_comm_type` |
| fused attention kernel | 在 GPU 上完成 attention 计算、mask、通信或通信 overlap |

L10 patch 不写通信。它只证明每个 rank 见到一系列 K/V chunk 后，online softmax 状态如何合并。

## 8. Causal mask、backward 和生产边界

本关测试的是无 causal mask 的 full attention。真实自回归训练需要 causal mask：query 位置只能看自己及之前的 key。分块后，mask 需要按 chunk 的序列范围判断：

1. 完全在 query 可见范围内的 chunk 正常累加。
2. 完全在未来的 chunk 可以跳过。
3. 部分重叠的 chunk 要在当前 `scores_chunk` 内填 `-inf`。

Backward 也有额外复杂度。训练中不能为了 backward 保存所有 chunk 的完整 score 矩阵，否则省下的显存会被反向传播拿回去。FlashAttention 类实现通常在 backward 中重算必要的分数，并用 forward 保存的 softmax 统计量恢复梯度。

本讲先只验证 forward 数学和 autograd 能跑通。生产系统还要看 dropout、mask、packed sequence、通信 overlap、TE 版本、stream、kernel layout 和数值精度。

## 9. RoPE / YaRN 和 ring attention 的边界

长上下文还涉及位置编码外推。RoPE 把位置写进 Q/K 的旋转角度里；YaRN 通过频率缩放和 attention temperature 补偿来缓解训练长度外推到更长上下文时的困境。

它们和 ring attention 解决的问题不同：

| 技术 | 解决的问题 |
|---|---|
| RoPE | 让模型感知 token 的相对位置 |
| YaRN | 让 RoPE 在超过训练上下文时更稳 |
| Online softmax / FlashAttention | 降低 attention 中间激活和内存访问压力 |
| Context Parallel | 把序列维切到多个 rank，分摊激活和 K/V 持有压力 |

`eval_yarn.py` 只输出简化数学估算。真实是否能支持 32K 或 128K，要看长上下文 validation perplexity、needle 测试、训练稳定性和 serving 时的 KV cache 压力。

## 10. Patch 验收什么

Patch 只要求实现：

```python
ring_attention_forward(q, k, v, num_chunks=1)
```

验收合同是：

1. `num_chunks=1` 与 PyTorch SDPA 对齐。
2. `num_chunks=4` 与 SDPA 对齐。
3. `Sk` 不能整除 chunk 数时仍正确。
4. 长 K/V 场景下数值误差在容差内。
5. autograd backward 能产生有限梯度。

它没有验收真实 CP 通信、causal mask、dropout、custom backward、GPU kernel 和性能。通过 patch 后，你只能声明 online softmax 的 forward 主路径正确；系统可用性还要看后续 drill 和集群验证。

## 11. 生产排查顺序

遇到长上下文 OOM 或 CP 结果异常时，按下面顺序查：

1. Shape：确认 `B,H,Sq,Sk,D` 和 dtype。
2. Score 激活：估算 `B * H * Sq * Sk * dtype_bytes`。
3. Chunk：检查 `num_chunks`、chunk_len、uneven chunk 和峰值 `scores_chunk`。
4. Online softmax：核对 max、denom、out 的缩放是否一致。
5. CP group：确认 `context_parallel_size`、rank group 和 global ranks。
6. Mask：确认 causal / padding / packed sequence 的可见范围。
7. Communication：检查 ring 步数、通信字节、stream 和 overlap。
8. Validation：用 SDPA、短序列、长序列、perplexity 或 needle 测试分层验证。

这个顺序能把数学错误、shape 错误、通信错误和评估证据分开。

## 12. 小结

L10 的主线是：full attention score 矩阵会让长上下文训练的激活显存按二次方增长；online softmax 让分块 attention 在不保存完整 score 矩阵的情况下得到等价输出；Context Parallel 把 K/V chunk 分散到多个 rank，并用 ring 通信让每个 rank 见到完整上下文。

完成 patch 后，你应该能解释每个 chunk 如何更新 `running_max`、`running_denom` 和 `running_out`，并能把 CPU 上的分块循环对应到 Megatron/TE 的 CP 配置和通信边界。
