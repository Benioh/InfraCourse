# L20：vLLM Serving Baseline and Typical-p Sampling

推理服务上线前，先要建立 baseline：请求能不能打到服务、端口是否打开、模型名是否一致、TTFT 和吞吐有没有被记录。L20 在这个 baseline 上增加一个采样 patch：实现 typical-p logits filter，理解它和 top-p/top-k 的不同排序依据。

## 0. 学完后要能做什么

- 说明 OpenAI-compatible request 到 LLMEngine、sampler、metrics 的链路。
- 区分 TTFT、ITL、requests/sec 和 validation_only 状态。
- 解释 typical-p 的 surprisal、entropy、distance 和 probability mass cutoff。
- 实现 `typical_p_filter(logits, typical_p, filter_value)`。
- 读懂 vLLM sampler 中 temperature、logits processors、top-k/top-p 的位置。

## 1. Serving baseline：先记录服务事实

`run_vllm_lab.py` 不假装本地一定有 vLLM server。它会检查 `vllm` 包是否可 import、端口是否打开、应执行的 serve 命令是什么。如果端口没开，run 仍会写 `validation_only` artifact，说明这次只验证了环境和配置，没有产生真实 serving 指标。

真实 benchmark 必须记录 workload 条件：模型、端口、max model len、prompt 长度、output 长度、并发和采样参数。没有这些条件，TTFT 或 requests/sec 无法和另一轮实验比较。

## 2. Sampling 过滤器的位置

LLM forward 输出 logits 后，sampler 会做一系列处理：温度缩放、logits processors、top-k/top-p 等过滤，然后从剩余概率分布里采样。typical-p 是同类过滤器，但排序依据不同。

top-p 按 token 概率从大到小累计。typical-p 先计算每个 token 的 surprisal：

```text
surprisal(token) = -log(prob(token))
```

再计算 entropy，也就是当前分布的期望 surprisal：

```text
entropy = sum(prob * surprisal)
```

typical-p 保留 `abs(surprisal - entropy)` 小的 token，直到累计概率质量达到阈值。直觉上，它保留信息量接近当前分布期望的 token，过滤过于普通或过于意外的 token。

## 3. Patch 实现

`typical_p_filter` 的输入是 `(B, V)` logits，输出 shape 不变。实现步骤：

1. `probs = softmax(logits.float(), dim=-1)`。
2. `info = -log(probs + eps)`。
3. `entropy = (probs * info).sum(dim=-1, keepdim=True)`。
4. `dist = abs(info - entropy)`。
5. 按 `dist` 升序排序，并 gather 对应概率。
6. 对 sorted probs 做 cumsum，超过 `typical_p` 的位置标记删除。
7. 强制第 0 个位置保留，避免极小阈值删除所有 token。
8. scatter 回原 vocab 顺序，并用 `filter_value` mask。

`typical_p >= 1.0` 时应原样返回 logits。`typical_p` 很小时也至少保留一个 token。

## 4. 和 vLLM 源码对照

vLLM 的 `SamplingParams` 定义温度、top-p、top-k、min-p 等参数，并在 greedy 模式下把 top-p/top-k/min-p 归到禁用状态。`Sampler.sample` 里先处理 greedy，再应用 temperature、argmax-invariant processors，最后调用 top-k/top-p sampler。

L20 的 patch 没有直接改 vLLM 源码，而是用一个最小函数训练采样过滤器的思维：过滤器的输入输出保持 logits shape；排序、累计和 scatter 都必须按 batch row 独立执行。

## 5. 验收和复盘

patch-test 覆盖 5 个边界：

| 测试 | 验收点 |
|---|---|
| `test_typical_p_one_keeps_all` | `typical_p=1.0` 原样返回 |
| `test_typical_p_zero_keeps_one` | 极小阈值仍保留 1 个 token |
| `test_filter_value_applied` | 被删位置等于 `filter_value`，保留位置不变 |
| `test_kept_tokens_dist_minimal` | 保留 token 的 distance 不大于被删 token |
| `test_works_with_batch` | batch 每行独立处理，至少保留 1 个 token |

smoke 命令：

```bash
python labs/l19_vllm_serving_baseline/scripts/run_vllm_lab.py --run-id l20_validation
```

如果没有本地 server，报告会标记 `validation_only`。这不是 serving 性能结果，但它仍然是有用证据：环境、配置、命令和端口状态都被记录下来。

---

## 补充：Serving Pipeline 全链路

### 1. Request 生命周期

一个推理请求从到达到返回的完整链路：

```
Client Request
  → HTTP Server 接收（FastAPI/aiohttp）
  → Tokenize（text → token_ids）
  → 加入 Scheduler 等待队列
  → Scheduler 调度（分配 KV cache blocks）
  → Prefill 阶段（处理完整 prompt，compute-bound）
  → Decode 阶段（逐 token 生成，memory-bound）
  → Detokenize（token_ids → text）
  → 流式/完整返回 Response
```

每个环节都有延迟贡献，优化 serving 需要识别哪个环节是瓶颈。

### 2. LLMEngine 架构

vLLM 的 `LLMEngine` 是核心调度引擎，由三个主要组件构成：

- **Scheduler**：决定哪些 request 在当前 step 执行。维护 waiting/running/swapped 三个队列。每步根据可用 KV cache blocks 决定 admit 新请求还是继续已有请求。
- **Model Runner**：负责实际的模型 forward。管理输入 tensor 的拼接（多个请求 batch 在一起）、position encoding、attention mask 构建。
- **Cache Manager（Block Manager）**：管理 GPU 显存中的 KV cache block 池。负责 block 分配、引用计数、copy-on-write（用于 beam search）、eviction（用于 prefix caching）。

三者协作流程：Scheduler 选出本步请求 → Block Manager 分配/确认 blocks → Model Runner 执行 forward → 输出 token 返回 Scheduler 更新状态。

### 3. Prefill vs Decode：两种截然不同的计算特征

| 特征 | Prefill | Decode |
|---|---|---|
| 输入 token 数 | 多（整个 prompt） | 1（上一步生成的 token） |
| 计算特征 | Compute-bound（大矩阵乘） | Memory-bound（小矩阵乘，大量 KV cache 读取） |
| GPU 利用率 | 高（接近峰值 FLOPS） | 低（受限于显存带宽） |
| 延迟贡献 | TTFT | ITL（每个 token 的延迟） |
| Batch 效率 | 高（大 sequence 并行） | 依赖 batch size |

这种差异是 splitwise/disaggregated serving 的动机：用不同硬件分别优化 prefill 和 decode。

### 4. Continuous Batching vs Static Batching

**Static Batching（传统方式）**：
- 收集 B 个请求组成一个 batch，等所有请求生成完毕（或达到 max_length）才释放。
- 短请求必须等长请求完成，GPU 利用率随时间下降（padding 浪费）。

**Continuous Batching（vLLM 方式）**：
- 每个 decode step 后检查：已完成的请求立即移出 batch，空出的 slot 立即插入新请求。
- 优势：GPU 始终在做有用计算，throughput 可提高 2-10×。
- 实现要求：Scheduler 必须逐 step 决策，KV cache 管理必须支持动态分配/释放。

vLLM 的 iteration-level scheduling 就是 continuous batching 的实现。

### 5. KV Cache 管理：Block 分配、复用与驱逐

vLLM 的 PagedAttention 把 KV cache 组织成固定大小的 block（默认 16 tokens/block）：

- **Block 分配**：新请求到来时，按需分配 block（不预分配 max_length），避免显存浪费。
- **Block 复用（Prefix Caching）**：如果多个请求共享相同 prefix（如 system prompt），对应的 KV cache blocks 可以共享（引用计数 > 1），不重复计算。
- **Copy-on-Write**：beam search 中多个 beam 共享前缀 blocks，分叉时才复制。
- **驱逐（Eviction）**：prefix cache 满时按 LRU 或 priority 驱逐最旧的 cached blocks。
- **Swap**：GPU block 不足时，把低优先级请求的 blocks swap 到 CPU，腾出空间给新请求 prefill。

Block 粒度管理使得显存碎片极低（类比 OS 的虚拟内存分页），这是 vLLM 高吞吐的核心。

### 6. TTFT vs ITL：关键性能指标

**TTFT（Time To First Token）**：从请求到达到第一个 output token 生成的时间。

影响因素：
- Prompt 长度（prefill 计算量）
- 排队时间（scheduler 拥塞）
- KV cache 命中率（prefix caching 是否生效）

**ITL（Inter-Token Latency）**：相邻两个 output token 之间的生成间隔。

影响因素：
- 当前 batch size（decode 阶段的并发请求数）
- KV cache 读取带宽（sequence 越长，每步读取越多）
- GPU 显存带宽利用率

**优化取舍**：
- 减小 batch size → ITL 下降但 throughput 也下降
- 增大 batch size → throughput 上升但 ITL 上升
- Chunked prefill → TTFT 略增但 decode 请求的 ITL 不被 prefill 抢占
- Prefix caching → TTFT 显著下降（跳过重复 prefill）

生产系统通常设定 SLO（如 TTFT < 500ms, ITL < 50ms），scheduler 据此做 admission control。
