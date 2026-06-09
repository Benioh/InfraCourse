# L03.5 讲义：Debug 方法论

## 1. 学完要能回答什么

1. 遇到一个训练 bug，正确的第一步是什么？（不是改代码，是缩小范围）
2. "最小复现"具体要缩小哪些维度？为什么？
3. Baseline 应该怎么选？为什么不能跳过 baseline 直接对比两个都可能出错的版本？
4. Tensor 对齐时，为什么要逐层比而不是只比最终 loss？
5. 多卡 hang 住时，第一步应该看什么？
6. 显存 OOM 的排查路径？
7. Loss 对不上时如何区分是数据问题还是模型问题？
8. 什么是"好的 bug report"？需要包含哪些信息？

## 2. 第一步：最小复现

### 2.1 为什么最小复现是第一步

在复杂系统中，问题可能来自任何层：数据、模型、框架、硬件、配置。如果你在完整配置（64 卡、长序列、full dataset）下 debug，每次实验要跑几小时，你一天只能试 2-3 个假设。

最小复现的目标：把实验周期从小时缩短到分钟，同时保持问题可复现。

### 2.2 缩小维度

| 维度 | 完整配置 | 最小复现 | 理由 |
|---|---|---|---|
| 节点数 | 8 节点 | 1 节点 | 排除网络和跨节点通信问题 |
| GPU 数 | 8 卡 | 1-2 卡 | 排除分布式并行问题 |
| Batch size | 32 | 1-2 | 减少内存和计算，加快迭代 |
| Sequence length | 8192 | 128-512 | 排除长序列特有问题 |
| 训练步数 | 10000 | 1-10 | 快速验证 |
| 随机种子 | 随机 | 固定 | 消除随机性，使问题稳定复现 |
| Dataset | 完整数据集 | 1-10 条固定样本 | 排除数据本身的问题 |

### 2.3 固定随机种子

```python
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 注意：这会降低性能，只在 debug 时使用
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**注意**：即使固定种子，某些 CUDA op 仍有非确定性（如 atomicAdd）。使用 `torch.use_deterministic_algorithms(True)` 可以强制确定性，但某些 op 会 fallback 到慢路径或报错。

### 2.4 最小复现的验收标准

- 问题能稳定复现（10 次跑 10 次都出问题）
- 单次实验时间 < 5 分钟
- 只用 1 台机器 + 1-2 卡

如果缩小后问题消失了，这本身就是信息——问题可能和规模/并行相关。

## 3. 第二步：建立 Baseline

### 3.1 什么是好的 Baseline

Baseline 是一个你**确信正确**的版本，用来和出问题的版本对比。

好的 baseline 选择：
| 场景 | 推荐 Baseline |
|---|---|
| 自定义模型 vs HF | HF transformers 的原始模型 |
| 多卡 vs 单卡 | 单卡版本（去掉所有并行） |
| FlashAttention vs Eager | eager attention（数学上一定正确） |
| Packed sequence | 未 packed 版本（每条样本独立跑） |
| FSDP/TP/PP | 单卡 DDP 或纯单卡 |
| 自定义 SFT template | 最简 prompt + response 模板 |

### 3.2 单变量原则

每次对比只改一个变量。如果你同时从"单卡 + eager + 未 packed"切到"多卡 + flash + packed"，出了问题你不知道是哪个变化导致的。

正确路径：
```
baseline: 单卡 + eager + 未 packed ✓
step 1:   单卡 + flash + 未 packed  → 找 attention 问题
step 2:   单卡 + flash + packed    → 找 pack 问题
step 3:   多卡 + flash + packed    → 找并行问题
```

## 4. 第三步：Tensor 对齐

### 4.1 为什么要逐层对比

只比较最终 loss 有两个问题：
1. 小的数值差异经过 30 层 transformer 会被指数级放大。
2. 你无法定位是哪一层开始出错。

逐层对比能告诉你："第 5 层的 attention output 开始偏离 baseline"——这就能精确定位到具体组件。

### 4.2 对比路径

```
input_ids (应该完全一致)
    ↓
embedding output (应该完全一致)
    ↓
layer 0 attention output (应该很小误差或完全一致)
    ↓
layer 0 MLP output
    ↓
...
    ↓
layer N hidden states
    ↓
logits (误差可能已经放大)
    ↓
loss (如果前面有误差，这里会更大)
```

### 4.3 对比指标

```python
def tensor_diff_report(a: torch.Tensor, b: torch.Tensor) -> dict:
    abs_diff = (a - b).abs()
    rel_diff = abs_diff / (b.abs() + 1e-8)
    cosine_sim = torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0),
        b.flatten().unsqueeze(0)
    ).item()

    return {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "cosine_similarity": cosine_sim,
    }
```

**判断标准**：
- `cosine_sim > 0.9999` → 基本一致（FP16 数值误差范围内）
- `cosine_sim < 0.99` → 有实质性差异，需要调查
- `cosine_sim < 0.9` → 严重偏离，前面几层一定有 bug

### 4.4 常见对齐陷阱

| 层 | 常见问题 |
|---|---|
| embedding | 不同 tokenizer 版本、vocab size 不同 |
| attention | mask 不对（packed 场景）、position_ids 不对（RoPE） |
| MLP | 某些框架用 gated MLP、某些不是 |
| logits | hidden_size 最后一维做 layernorm 的位置不同 |
| loss | reduction 方式（mean vs sum）、mask 覆盖范围 |

## 5. 第四步：常见问题模式

### 5.1 Loss 对不上

**排查流程**：
1. 先对齐 input_ids（确认数据一致）
2. 检查 loss mask / labels：prompt 部分是否标记为 -100？
3. 检查 reduction：是对所有 token 做 mean 还是只对非 -100 的做 mean？
4. 检查 loss scale：是否除以了 micro_batch 或 gradient accumulation steps？

### 5.2 Packed 后效果崩

**排查流程**：
1. 检查 attention_mask：packed 的多条样本之间是否正确设置了 causal mask + 样本隔离？
2. 检查 position_ids：每条样本的 position_ids 是否从 0 开始重置？
3. 检查 loss mask：确认 prompt token 和 pad token 都被 mask 掉。
4. 如果用 varlen attention：确认 cu_seqlens 正确。

### 5.3 多卡 Hang

**排查流程**：
1. 设置 `NCCL_DEBUG=INFO` 看日志。
2. 检查是否所有 rank 都到达了同一个 collective（all-reduce/all-gather）。
3. 如果只有部分 rank 到达 → 某些 rank 在之前的逻辑中走了不同分支。
4. 常见原因：某些 rank 的 batch 为空、条件判断里有只在部分 rank 满足的分支。

### 5.4 多卡比单卡慢

**排查流程**：
1. nsys 看通信是否和计算 overlap。
2. 检查 batch 在 rank 间是否均衡。
3. 检查是否有 straggler（某个 rank 的 backward 特别慢）。
4. 检查网络带宽（InfiniBand vs Ethernet）。

### 5.5 显存异常

**排查流程**：
1. `torch.cuda.memory_summary()` 看当前分配。
2. memory profiler 看历史分配。
3. 检查是否有 tensor 被意外保留（在 list 中、作为 attribute、在闭包中）。
4. 检查是否保存了计算图（`retain_graph=True` 或者 loss 被保留）。

### 5.6 时间测量不准

**检查**：是否在测量前后加了 `torch.cuda.synchronize()`？没有 sync 的时间测量对 GPU 操作无意义。

## 6. 好的 Bug Report 模板

```markdown
## 问题描述
一句话说明现象。

## 复现步骤
- 配置：xxx
- 命令：xxx
- 是否稳定复现：是/否

## 期望行为
应该看到什么。

## 实际行为
实际看到什么（贴具体数字或日志）。

## 已尝试的排查
- 缩小到几卡：___
- 固定 seed：___
- baseline 是什么：___
- tensor 对齐到哪一层：___

## 环境
- GPU 型号/数量
- PyTorch 版本
- 框架版本
- CUDA/NCCL 版本
```

## 7. 小结

| 步骤 | 一句话 |
|---|---|
| 最小复现 | 缩小到 1 卡、小 batch、短 seq、固定 seed、1 step，使问题稳定复现 |
| 建立 baseline | 找一个确信正确的版本，和出问题的版本做单变量对比 |
| Tensor 对齐 | 逐层比 cosine similarity，找到第一个 diverge 的层 |
| 模式匹配 | 根据 diverge 层和症状，匹配已知问题模式 |
