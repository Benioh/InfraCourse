# L10 · 长上下文 Context Parallel：Online Softmax 与 Ring Attention

这一讲解决长上下文训练里 attention 激活显存过高的问题。学生先在单进程 CPU 上实现 `ring_attention_forward(q, k, v, num_chunks)`：把 K/V 沿序列维切块，逐块维护 online softmax 的 `running_max`、`running_denom` 和 `running_out`，最后得到与 full attention 数学等价的输出。然后把这个本地 chunk 循环对照到 Megatron / Transformer Engine 的 Context Parallel。

## 学习路线

建议按下面顺序走，先讲清长上下文瓶颈和 online softmax，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L10 在训练系统、长上下文和 Megatron CP 路线里的位置。
2. 读 [lecture.md](lecture.md)：理解 score 矩阵显存、Q/K/V shape、online softmax、K/V chunk、CP ring 和 RoPE/YaRN 边界。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra、Megatron/TE CP 和 drill 脚本读源码。
4. 跑 notebook：`notebooks/n13_rope_yarn_math.ipynb` 和 `notebooks/n14_context_parallel_ringattn.ipynb`。
5. 做 quiz：确认 online softmax 等价性、显存估算、ring 通信和 causal mask 边界。
6. 做 patch：实现 `ring_attention_forward`，并通过 5 个 CPU 测试。
7. 跑 drill：用 `run_seqlen.py` 和 `eval_yarn.py` 记录序列长度、CP size、通信字节和 YaRN 估算。
8. 填写 [outputs/long_context_template.md](outputs/long_context_template.md)，沉淀长上下文复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Training systems / Long context / Context Parallel |
| 它承接什么 | L09 的训练 loop 和 scheduler 证据链 |
| 它解决什么问题 | attention score 激活按 `Sq * Sk` 增长，长上下文训练容易 OOM |
| 它连接哪些指标或证据 | `seq_len`、`cp_size`、`attn_comm_bytes`、`peak_mem_gb_cp`、tokens/sec 估算、SDPA 数值差 |
| 它连接哪些源码 | patch starter/reference/tests、MiniInfra ring plan、Megatron `ModelParallelConfig`、parallel_state、TE DotProductAttention |
| lab 检验什么 | 分块 attention 的 online softmax 累加与 full attention 输出等价 |

## 你会学到什么

- 为什么 full attention 的 score 激活会按 `B * H * Sq * Sk` 增长。
- Q/K/V 的 `(B,H,S,D)` shape 如何映射到 `scores` 和输出。
- online softmax 为什么必须同时重缩放旧的分母和旧的未归一化输出。
- `num_chunks=1`、`num_chunks=4`、uneven chunks 和长 K/V 为什么应得到同一 attention 语义。
- Context Parallel 中 Q stationary、K/V ring、CP group 和 TE `cp_comm_type` 分别在做什么。
- RoPE / YaRN 在长上下文中负责位置外推，和 ring attention 的显存路径不是同一个问题。

## Patch 闭环

```bash
cat labs/l09_long_context_cp/patch/task.md
$EDITOR labs/l09_long_context_cp/patch/starter/ring_attention.py
make patch-test M=l09_long_context_cp
```

测试覆盖：

| 测试 | 验证 |
|---|---|
| `test_matches_full_attention_one_chunk` | 单块输出与 PyTorch SDPA 对齐 |
| `test_matches_full_attention_four_chunks` | 多块 online softmax 与 SDPA 对齐 |
| `test_handles_uneven_chunks` | `Sk=7, num_chunks=2` 的不均匀 chunk |
| `test_long_seq` | `Sq=256, Sk=1024, num_chunks=8` 的长 K/V |
| `test_grads_are_continuous` | autograd backward 能产生有限梯度 |

## Drill 闭环

```bash
python labs/l09_long_context_cp/scripts/run_seqlen.py --seq 16384 --cp 4
python labs/l09_long_context_cp/scripts/eval_yarn.py --train-ctx 4096 --eval-ctx 32768
```

`run_seqlen.py` 给出教学版 ring plan、通信字节和 CP 后峰值显存估算。`eval_yarn.py` 只做 RoPE 外推的数学估算，不代表真实长上下文 perplexity。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查长上下文 OOM、chunk 数值不一致、CP shape mismatch 和 ring 通信证据 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 online softmax、MiniInfra、Megatron CP 和 TE attention 主路径 |
| [outputs/long_context_template.md](outputs/long_context_template.md) | 记录一次长上下文实验的形状、显存估算、通信和验证结论 |

## 进入下一讲

`make patch-test M=l09_long_context_cp` 通过，并完成一次 `run_seqlen.py` 复盘后，进入 [L11 Megatron 预训练生命周期](../l10_megatron_pretrain_lifecycle/README.md)。下一讲会把训练主线接回更完整的 Megatron lifecycle。
