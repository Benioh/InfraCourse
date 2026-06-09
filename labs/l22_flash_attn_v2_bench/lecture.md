# L23：PyTorch SDPA 与 FlashAttention benchmark

## 0. 本讲目标

- 理解 eager attention 为什么会显式产生 `[B, H, T, T]` 的 score 和 weight。
- 能解释 FlashAttention/SDPA 如何通过分块和在线 softmax 降低 HBM 流量和峰值显存。
- 能判断 PyTorch `scaled_dot_product_attention` 什么时候可能走 flash、efficient、math 或 cudnn backend。
- 能设计一组最小 benchmark，分别验证数值、耗时、speedup 和峰值显存。
- 能把 patch 测试、drill artifact 和真实项目里的 SDPA wrapper 对上。

## 1. 问题背景：长序列 attention 为什么先卡在内存

在 Transformer attention 中，输入通常按 `[B, H, T, D]` 组织：

- `B` 是 batch size。
- `H` 是 attention head 数。
- `T` 是序列长度。
- `D` 是每个 head 的维度。

eager attention 的直接写法是：

```python
scores = (q @ k.transpose(-2, -1)) / sqrt(D)
weights = scores.softmax(dim=-1)
out = weights @ v
```

这段代码清楚，但会创建 `[B, H, T, T]` 的 `scores` 和 `weights`。当 `T=4096`、`H=16` 时，一个 batch 的 score 矩阵已经有 `16 * 4096 * 4096` 个元素。bf16 下仅 score 就接近 512 MiB，实际路径还可能同时保留 mask、weights、临时 buffer 和 autograd 状态。

推理服务里，这个问题会表现在两个地方：

1. prefill 处理长 prompt 时，attention 一次看到完整上下文，`T^2` 中间矩阵很快放大。
2. decode 虽然每步 query 很短，但 KV cache 很长，attention backend 仍然决定每步读取历史 KV 的代价。

本讲不把 FlashAttention 当成一个黑盒库名。我们先用 eager baseline 暴露成本，再用 PyTorch SDPA 观察统一 API 下的 backend 行为。

## 2. 核心概念

### 2.1 Scaled Dot-Product Attention

**定义：**
给定 query、key、value，attention 先计算 `QK^T / sqrt(D)`，经过 mask 和 softmax 得到权重，再乘以 `V` 得到输出。

**输入：**
`q`、`k`、`v` 的 shape 都是 `[B, H, T, D]`。如果是 causal attention，第 `i` 个 token 只能看 `j <= i` 的位置。

**中间状态：**
eager baseline 会显式保存 `scores` 和 `weights`，shape 是 `[B, H, T, T]`。

**输出：**
输出 shape 回到 `[B, H, T, D]`。

**代价和边界：**
计算量仍然随 `T^2` 增长。eager 写法额外把 `T^2` 中间矩阵写入和读出 HBM，长序列下经常先受内存带宽和峰值显存限制。

### 2.2 FlashAttention / 在线 softmax

**定义：**
FlashAttention 类 kernel 将 QK、softmax 和 AV 分块融合，在片上 SRAM/register 中维护每个 query 行的 running max、running sum 和部分输出，避免把完整 `[T, T]` attention matrix 写到 HBM。

**直观理解：**
普通 eager 写法像先把整张表算出来，再逐行归一化。FlashAttention 像一块一块读 key/value，每读一块就更新当前行的最大值、归一化分母和输出累积。为了数值稳定，running sum 必须在最大值变化时重新缩放。

**输入：**
同样是 `q/k/v`、causal 标记、scale、dtype 和 device。

**中间状态：**
不保存完整 score/weight 矩阵。每个 tile 维护局部分数、局部最大值、归一化分母和输出累积。

**输出：**
输出仍然是 `[B, H, T, D]`，数值应与 eager baseline 在容差内一致。

**代价和边界：**
它没有把 attention 的理论计算量改成线性。收益来自减少 HBM 读写和中间显存。它也依赖硬件、dtype、head_dim、mask、dropout 和 backend 可用性；条件不满足时会 fallback。

### 2.3 PyTorch SDPA

**定义：**
`torch.nn.functional.scaled_dot_product_attention` 是 PyTorch 提供的统一 attention API。调用方传入 `q/k/v`、`attn_mask`、`is_causal`、`dropout_p`、`scale` 和 `enable_gqa` 等参数，PyTorch 决定后端实现。

**直观理解：**
SDPA 像一个 attention 入口。学生代码只调用一个函数，但真实执行可能是 math、memory efficient、flash 或 cudnn attention。

**为什么要学：**
生产代码经常只留下一个 SDPA 调用。排查性能时，必须知道 API 层和 kernel 层之间还有 dispatch 条件。否则看到 `scaled_dot_product_attention` 就直接断言用了 FlashAttention，会得到错误结论。

**常见边界：**

- CPU smoke 通常走 math backend。
- fp32 在很多 GPU 路径上会走 math 或非 flash backend。
- head_dim、mask 形态、dropout、GQA 和设备能力会影响选择。
- 同时传复杂 `attn_mask` 和 `is_causal` 时要确认语义，避免重复 mask 或 fallback。

## 3. 机制链路：从 eager baseline 到 benchmark artifact

本讲的最小系统可以拆成五步。

### 3.1 先写 eager baseline

输入是 `q/k/v`。中间先计算：

```python
scale = 1.0 / math.sqrt(q.size(-1))
scores = (q @ k.transpose(-2, -1)) * scale
```

如果 `causal=True`，baseline 构造上三角 mask，把未来位置填成 `-inf`，然后做 softmax。输出是 `weights @ v`。这一步的价值在于给 SDPA 一个可解释、可对照的数学基线。

### 3.2 再写 SDPA 调用

`flash_attention()` 在 patch 里只做一件事：

```python
return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
```

这个函数名保留了课程里的 `flash_attention`，但实现使用 PyTorch SDPA。课堂里要明确：这个 wrapper 不保证每次都走 flash backend。它让我们用同一 API 在 CPU smoke 和 CUDA benchmark 中跑通验证。

### 3.3 正确性和性能分开测

正确性使用 fp32 和较小 `seq_len`。原因是：

- fp32 更适合和 eager baseline 做严格容差比较。
- 小序列避免 CPU smoke 或低显存机器被 `[T, T]` baseline 压垮。

性能使用配置指定的 dtype 和完整 `seq_len`。原因是：

- FlashAttention 类 kernel 的收益通常出现在 fp16/bf16、CUDA、较长序列。
- CPU 或短序列的调度开销可能盖过内核差异。

### 3.4 计时必须同步

CUDA kernel 是异步 launch。计时前后必须调用 `torch.cuda.synchronize()`，否则测到的是提交开销，不是 kernel 完成时间。本讲 reference 的 `_time_iters()` 把 synchronize 包在循环前后，返回每次调用的平均毫秒数。

### 3.5 artifact 要能复盘

`scripts/run_bench.py` 会读取 config，逐个 shape 运行 `bench_attention()`，落盘三类证据：

- `metrics.jsonl`：每个 shape 的结构化指标。
- `artifacts/bench.csv`：便于快速查看的表格。
- `artifacts/bench_summary.json`：包含 `accept` 和全部 rows。

没有 shape、dtype、device、iters、causal 和 max_abs_diff 的记录时，speedup 数字不能迁移到真实项目。

## 4. MiniInfra 和真实源码怎么连起来

MiniInfra 的 `triton_softmax.py` 展示了在线 softmax 的核心数学。它用 Python 列表维护 `running_max` 和 `running_sum`，把一行拆成多个 block。这个例子不证明 GPU 性能，它只帮助学生理解分块 softmax 如何保持数值等价。

真实项目里的 SDPA wrapper 会多出几类生产复杂度：

- layout：很多模型内部是 `[B, T, H, D]`，SDPA 需要 `[B, H, T, D]`，所以调用前后要 transpose。
- dtype：query、key、value 的 dtype 要对齐，否则 SDPA 会报错或触发转换。
- cache：serving decode 会从 KV cache 中取 key/value，再按请求切片。
- mask：decoder、encoder-only、cross attention、sliding window、ALiBi 会改变 mask 选择。
- backend：有些代码用 `sdpa_kernel()` 约束或排序候选 backend。

因此，patch 只保留最小合同。它让学生掌握 attention 公式、SDPA 调用和指标记录。读真实源码时，要继续追 layout、cache、mask 和 backend 约束。

## 5. Benchmark 怎么解释

性能结论至少包含五个字段：

| 字段 | 为什么需要 |
|---|---|
| 比较对象 | eager baseline 还是另一个 kernel |
| 指标 | time、speedup、peak memory、max_abs_diff 分别回答不同问题 |
| 测试条件 | device、dtype、shape、causal、iters、warmup 决定结论边界 |
| 优化原因 | 是否减少 `[T, T]` 中间矩阵和 HBM 访问 |
| 代价 | backend 条件、数值容差、mask 支持、短序列收益不稳定 |

CPU smoke 通过时，只能说：公式、causal mask、SDPA API 和 artifact 写入路径是通的。CUDA benchmark 通过时，才能讨论 flash backend 在该硬件和 shape 下的速度与显存趋势。

## 6. Lab 验收边界

本讲 patch 只验收三个函数：

1. `eager_attention()`：显式实现 `QK^T / sqrt(D)`、causal mask、softmax 和 `weights @ v`。
2. `flash_attention()`：调用 PyTorch SDPA，并传入 causal 语义。
3. `bench_attention()`：分开做 fp32 正确性和目标 dtype 性能，返回规定指标。

测试覆盖：

- shape 是否保持 `[B, H, T, D]`。
- causal mask 是否阻止未来 token 泄漏。
- SDPA 输出是否与 eager baseline 在 fp32 容差内一致。
- 指标字典是否包含耗时、speedup、max_abs_diff 和峰值显存。
- CUDA 可用时，是否观察到速度和显存方向符合预期。

测试没有覆盖所有生产分支，例如 dropout、autograd、复杂 mask、GQA 的全部组合、具体 backend 强制选择、不同 GPU 架构差异。课堂里不要把 patch 通过解释成线上配置已经安全。

## 7. 排查路径

遇到 SDPA/FlashAttention 相关问题时，按这个顺序查：

1. 固定输入：记录 `B/H/T/D`、dtype、device、causal、mask、dropout、head_dim、PyTorch 版本和 GPU 型号。
2. 固定正确性：用小 shape 和 fp32 对比 eager baseline，先排除公式和 mask 问题。
3. 固定 backend：确认当前条件是否允许 flash backend，必要时用 profiler 或 backend 约束验证。
4. 固定计时：检查 warmup、iters、CUDA synchronize 和随机种子。
5. 固定显存：确认是否在 CUDA 路径调用 `reset_peak_memory_stats()` 和 `max_memory_allocated()`。
6. 固定 artifact：检查 `bench.csv`、`metrics.jsonl` 和 `bench_summary.json` 是否足以复现实验。

## 8. 小结

L23 的主线是：eager attention 暴露 `[T, T]` 中间矩阵成本；FlashAttention/SDPA 通过分块和在线 softmax 减少中间矩阵落到 HBM；PyTorch SDPA 通过统一 API 做 backend dispatch；benchmark 必须同时报告数值、耗时、显存和测试条件。掌握这条链后，学生才能在真实 serving 或训练项目里判断一个 attention 优化是否真的成立。

---

## 补充 A：FlashAttention v2 Tiling 算法详解

### A.1 核心思想

FlashAttention 的核心不是数学上改变 attention 公式——输出和 eager attention **完全一致**。它改变的是**内存访问模式**：把 O(T²) 大小的中间 score 矩阵从 HBM 移到 SRAM 中，分块计算。

### A.2 算法结构

```
外层循环: for each Q block (B_r rows of Q)
    内层循环: for each KV block (B_c columns of K/V)
        1. 从 HBM 加载 Q_block [B_r, d] 到 SRAM
        2. 从 HBM 加载 K_block [B_c, d], V_block [B_c, d] 到 SRAM
        3. 在 SRAM 中计算 S_block = Q_block @ K_block.T  [B_r, B_c]
        4. 在 SRAM 中做 online softmax update（不写回 HBM）
        5. 在 SRAM 中计算 O_block += softmax(S_block) @ V_block
    写回 O_block [B_r, d] 到 HBM
```

**关键点**：
- 中间矩阵 `S_block` 大小只有 `[B_r, B_c]`，放在 SRAM 中（通常 B_r = B_c = 128）。
- 完整 score 矩阵 `[T, T]` **从未在 HBM 中完整存在**。
- 内存复杂度从 O(T²) 降到 O(T)。

### A.3 Block Size 选择

```
B_r = min(d, SRAM_size / (4 * d))  # Q block rows
B_c = min(d, SRAM_size / (4 * d))  # KV block columns
```

H100 的 SM 有 228KB shared memory。对于 d=128：
- S_block [B_r, B_c] 需要 B_r × B_c × 4 bytes（FP32 用于 softmax 累积）
- 典型 B_r = B_c = 128 → S_block = 64KB

### A.4 FlashAttention v1 vs v2

| 方面 | v1 | v2 |
|---|---|---|
| 外层循环 | Over KV blocks | Over Q blocks |
| 并行度 | batch × heads | batch × heads × Q_blocks |
| GPU occupancy | 受限于 heads 数量 | Q blocks 提供更多并行 |
| 加速比 (vs eager) | 2-4× | 2× faster than v1 |

v2 把外层循环改成 Q blocks，让每个 Q block 独立处理所有 KV → 更好的 GPU occupancy。

### A.5 Backward 的挑战

Forward 没有保存完整的 softmax 权重矩阵（那是 O(T²)）。Backward 需要这个矩阵来计算梯度。

FlashAttention 的解决方案：**重新计算**。Backward 时再跑一遍 forward 的分块循环来恢复需要的中间值。这是用计算换内存的 trade-off，和 activation checkpoint 的哲学一致。

## 补充 B：FlexAttention（PyTorch 2.5+）

### B.1 问题

SDPA 支持的 mask 模式有限（causal、无 mask）。真实场景中 attention mask 可能非常复杂：
- Sliding window attention
- Prefix-based mask（前缀可见、后续 causal）
- Document-level mask（同 document 内可见，跨 document 不可见）
- Packed sequence 的样本隔离 mask

之前只能用 `attn_mask` tensor 传入，但这会绕过 FlashAttention 路径（flash 不支持任意 mask）。

### B.2 FlexAttention 的解决方案

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# 定义 score modification function
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# 创建 block mask（编译时确定哪些 block 需要计算）
block_mask = create_block_mask(causal_mask, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len)

# 使用 flex_attention
output = flex_attention(query, key, value, block_mask=block_mask)
```

### B.3 FlexAttention 的优势

1. **用户自定义 mask 逻辑**：用 Python 函数描述 mask，不需要 O(T²) 的 mask tensor。
2. **Block-sparse**：`create_block_mask` 在编译时确定哪些 128×128 的 block 全是 0（完全 masked），这些 block 直接跳过计算。
3. **融合进 kernel**：mask 判断在 Triton kernel 内部完成，不需要额外的 mask tensor 占用 HBM。
4. **支持 score modification**：不仅能 mask，还能修改 score（如 ALiBi position bias）。

### B.4 典型用途

```python
# Sliding window attention
def sliding_window(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() <= window_size

# Document mask (packed sequence)
def document_mask(b, h, q_idx, kv_idx):
    return document_id[q_idx] == document_id[kv_idx]

# Prefix + causal
def prefix_causal(b, h, q_idx, kv_idx):
    return (kv_idx < prefix_len) | (q_idx >= kv_idx)
```

## 补充 C：Varlen Attention（Packed Sequence）

### C.1 问题：Padding 浪费

传统 batch attention 要求同一 batch 内的序列等长。短序列需要 pad 到最长序列的长度。

```
Batch (pad to max_len=2048):
  seq 1: [tokens...](len=500)  + [PAD×1548]  ← 75% 浪费
  seq 2: [tokens...](len=1800) + [PAD×248]
  seq 3: [tokens...](len=200)  + [PAD×1848]  ← 90% 浪费
```

计算浪费：attention 在 pad token 上的计算完全无用（被 mask 掉），但 GPU 仍然做了 matmul。

### C.2 Varlen Attention 的解决方案

把所有序列**拼接**成一个长序列，用 `cu_seqlens`（cumulative sequence lengths）标记每条序列的边界：

```
Packed: [seq1_tokens | seq2_tokens | seq3_tokens]
         ←── 500 ──→←── 1800 ──→←── 200 ──→

cu_seqlens = [0, 500, 2300, 2500]  # 累积长度
max_seqlen = 1800                  # 最长序列长度
```

FlashAttention 的 `flash_attn_varlen_func` 接收这种格式：

```python
from flash_attn import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q,                  # [total_tokens, num_heads, head_dim]
    k, v,
    cu_seqlens_q,       # [batch_size + 1]
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=True,
)
```

### C.3 Varlen 的关键约束

1. **Attention 隔离**：seq1 的 token 只能 attend 到 seq1 的其他 token，不能跨序列。`cu_seqlens` 定义了隔离边界。
2. **Position IDs 重置**：每条序列的 RoPE position 从 0 开始，不是从 packed offset 开始。
3. **Loss mask**：只对非 pad、非 prompt 的 token 计算 loss。

### C.4 Varlen vs Padding 的性能

| 方面 | Padding | Varlen |
|---|---|---|
| 显存 | batch × max_len × hidden | sum(actual_lens) × hidden |
| 计算量 | batch × max_len² (attention) | sum(len_i²) (attention) |
| GPU 利用率 | 低（pad token 浪费） | 高（无浪费） |
| 实现复杂度 | 简单 | 需要正确维护 cu_seqlens |

当 batch 内序列长度差异大时（如 SFT 数据集），Varlen 可以节省 30-70% 的计算和显存。

### C.5 和 L28 SFT 的关系

L28 的 chat template 讲了 loss mask。当使用 packed sequence 时：
- `input_ids`：多条样本拼在一起
- `attention_mask`：需要确保跨样本不可见（varlen attention 自动处理）
- `position_ids`：每条样本从 0 开始
- `labels`：prompt 部分设为 -100，每条样本的 EOS 是否计入 loss 需要对齐

这三者（varlen attention + position_ids + loss mask）必须保持一致，否则就是 L03.5 Debug 方法论中"packed 后效果崩"的经典问题。
