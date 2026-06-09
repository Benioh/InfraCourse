# L07 讲义：Selective Activation Checkpoint

## 0. 本讲目标

学完这一讲，你应该能完成五件事：

- 解释 activation checkpoint 用什么换什么。
- 实现一个只包直接子模块的 `selective_checkpoint_wrap(model, policy_fn)`。
- 写出 `attention_only_policy(name, module)` 的匹配规则和边界。
- 说明 wrapper 为什么不能复制参数，以及 policy 全 False 为什么应是 no-op。
- 读懂 TorchTitan `parallelize_llama -> apply_ac` 的系统位置。

## 1. 真实问题：activation 往往先把训练顶爆

训练显存不只包括参数。L02 讲过参数、梯度和 optimizer state；真实 Transformer 训练还要在 forward 阶段保存许多中间 activation，供 backward 使用。activation 随 `micro_batch_size`、`seq_len`、`hidden` 和层数增长。

普通 autograd 的路径是：

```text
forward:
  compute layer output
  save backward needed tensors

backward:
  read saved tensors
  compute gradients
```

当保存的中间张量太多，forward 结束时还没开始释放它们，峰值显存就会很高。activation checkpoint 的核心交换关系是：

```text
少保存一部分 activation
backward 需要时重跑那一段 forward
```

它降低的是 activation 生命周期带来的峰值显存，代价是被包模块在一次训练 step 中多执行 forward。

## 2. activation checkpoint 与训练 checkpoint 的边界

两个名字很像，但系统位置不同。

activation checkpoint 是训练 step 内部的 autograd 技术。它影响 forward/backward 如何保存和重算中间状态。

训练 checkpoint 是磁盘 artifact。它保存模型参数、optimizer state、step、scheduler 等信息，供失败后 resume。

本讲 patch 实现的是 activation checkpoint wrapper。`run_torchtitan_stub.py` 里保存的 `checkpoint.pt` 是训练 checkpoint，用来演示框架边界；它不是本关 patch 的实现对象。

## 3. 本关最小接口

starter 提供两个函数：

```python
def selective_checkpoint_wrap(model, policy_fn) -> nn.Module:
    ...

def attention_only_policy(name: str, module: nn.Module) -> bool:
    ...
```

`policy_fn` 的输入是 child 名字和 child module，输出是 bool。`True` 表示这个 child 要被 `_CheckpointWrapper` 包住，`False` 表示保留原模块。

本关只遍历一层：

```python
for name, child in list(model.named_children()):
    if policy_fn(name, child):
        setattr(model, name, _CheckpointWrapper(child))
return model
```

这里特意用 `list(model.named_children())`，避免遍历过程中替换 child 导致 iterator 行为不清晰。它只处理 immediate children，不递归到孙子模块。真实项目可以递归，但本关的测试和讲义都围绕一层 child 语义。

## 4. `_CheckpointWrapper` 做了什么

`_CheckpointWrapper` 是一层薄外壳：

```python
class _CheckpointWrapper(nn.Module):
    def __init__(self, module):
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, use_reentrant=False, **kwargs)
```

它保存原 child 到 `self.module`。这意味着参数没有被复制，只是模块树多了一层 wrapper。`attn0.proj.weight` 可能在 state dict 名字里变成 `attn0.module.proj.weight`，但参数值和梯度路径仍来自原 child。

不要新建一个同结构模块来“替换”。复制参数会让 optimizer、FSDP wrapping、state dict 和 checkpoint resume 语义变得危险。训练框架里，参数对象身份和模块树结构都是系统状态的一部分。

## 5. 为什么输出和梯度仍应相同

假设模型是：

```text
attn0 -> mlp0 -> attn1 -> mlp1
```

`attention_only_policy` 会包住 `attn0` 和 `attn1`。forward 时，wrapped attention 的输出仍然传给下游模块。对下游来说，张量值应与裸模型一致。

差异发生在 autograd 保存策略上。checkpoint 内部尽量少保留 attention 的中间 activation。backward 走到 attention 时，PyTorch 用保存的边界信息重跑 attention forward，恢复反向需要的中间状态，然后继续计算梯度。

如果模块行为确定、输入相同、参数相同，那么重算得到的中间状态与原 forward 对齐，所以输出、输入梯度和参数梯度都应 allclose。本关 CPU 测试正是按这三个语义点验收。

有随机性的模块需要额外关注 RNG。dropout、随机采样或数据依赖控制流都可能让重算 forward 与原 forward 不一致。生产代码通常要显式处理 RNG、determinism check 和 debug 选项。

## 6. 为什么选择 attention

选择性 checkpoint 的价值在“省下多少 activation”和“多算多少 forward”之间取平衡。全量 checkpoint 省显存最多，但每个被包模块都要在 backward 重算。完全不用 checkpoint 计算少，但可能 OOM。

Transformer attention 常保存 Q/K/V、attention scores、softmax 相关中间状态；长序列下 attention activation 压力会很高。MLP 也有 activation，但 forward 计算通常更重。许多训练配置会优先对 attention 或特定 block 做 checkpoint，再根据 profiler 和 peak memory 调整。

本关 `attention_only_policy` 只是一个最小示例：

```python
n = name.lower()
return "attn" in n or "attention" in n
```

真实训练里，policy 可以按层号、模块类型、FQN、memory budget 或 profiler 结果决定。

## 7. TorchTitan 的系统位置

TorchTitan 的 Llama parallelize 路径里，顺序大致是：

```text
context parallel wrapping
tensor parallel parallelize
activation checkpoint apply_ac
torch.compile
FSDP
```

这个顺序很重要。activation checkpoint 改变模块 forward 的保存/重算边界；compile 和 FSDP 也会改写模块执行和参数分片。如果顺序混乱，可能遇到编译图缓存、FSDP wrapping、RNG 或状态字典问题。

TorchTitan 的生产 `apply_ac` 支持 full、selective、memory budget 等模式；本关的 wrapper 只保留最小 child-level policy。先把最小版本写对，再读 TorchTitan 复杂路径时就能看出每一层额外逻辑在解决什么问题。

## 8. Patch Tests 怎么读

运行：

```bash
make patch-test M=l06_torchtitan_training
```

测试含义：

| 测试 | 证明 |
|---|---|
| `test_output_matches_no_ckpt` | wrapped 模型输出与裸模型一致 |
| `test_grads_match_no_ckpt` | 输入梯度和参数梯度一致 |
| `test_attention_only_policy_counts` | policy 包住正确 child |
| `test_all_false_policy_is_noop` | 全 False policy 不改变直接 child 类型 |
| `test_memory_drops_with_attn_ckpt` | GPU 可用时 peak memory 下降 |

前四个测试 CPU 友好。第五个需要 CUDA；skip 表示没有验证条件，不表示已经证明显存下降。

## 9. Debug 路线

输出不等价：

- 检查 wrapper 是否调用 `checkpoint(self.module, ...)`。
- 检查是否复制了 child 或参数。
- 检查模型是否有随机层，RNG 是否一致。

梯度不等价：

- 检查 wrapped 模型和 base 模型是否从同一初始权重 clone。
- 检查参数是否仍在 module tree 中。
- 检查 policy 是否误包了不该包的 child。

policy 计数错误：

- 检查是否使用 `name.lower()`。
- 检查是否同时匹配 `attn` 和 `attention`。
- 检查是否只遍历 immediate children。

显存测试 skip：

- 记录为未验证。
- 不写成显存下降已证明。

## 10. 本讲小结

L07 讲的是训练 step 内部的 activation 显存控制。selective checkpoint 不改变模型数学目标，它改变 autograd 保存中间状态的方式：forward 少保存，backward 重算。最小实现只需要 wrapper、policy 和 in-place child 替换；真正难的是系统边界：参数不能复制，policy 要可替换，随机性要可控，GPU 显存收益要用 peak memory 和 step time 一起评估。

---

## 补充：Activation Checkpoint 的 Memory-Compute Tradeoff 深入分析

### 1. 定量分析：AC 节省多少显存

Activation checkpoint 的显存节省可以用公式估算：

```
saved_memory = activation_per_layer × num_checkpointed_layers
```

其中 `activation_per_layer` 取决于 hidden_size、sequence_length 和 batch_size。对于 Transformer，单层激活大约为 `2 × batch × seq × hidden × (10 + 24/tp)`（含 attention scores 和中间 MLP 激活）。如果对所有 N 层做 checkpoint，forward 只保留每层的输入 tensor，激活峰值从 O(N) 降到 O(1)（加上单层重算时的临时激活）。

### 2. 计算开销：重算的代价

Full recomputation 的额外计算量约为 **33%**。原因：正常训练一次 forward + 一次 backward；加 AC 后变成一次 forward（不保存）+ backward 中逐层 re-forward + backward。re-forward 的开销约等于原始 forward，而 backward 本身约 2× forward，因此总计从 3× forward 变为 4× forward，增加 ~33%。

### 3. Selective AC：只 checkpoint 最耗显存的部分

并非所有子模块的激活占用相同。Self-attention 的激活（尤其 QKV projection 和 attention score matrix `[batch, heads, seq, seq]`）通常占单层激活的大头。Selective AC 的策略：

- **只 checkpoint attention**：保留 MLP 激活（重算便宜但占显存少），重算 attention（占显存多但计算也多）。
- **只 checkpoint MLP**：保留 attention 激活，重算 MLP（compute 开销低，因为 MLP 是纯矩阵乘）。
- **op-level selective**：只重算特定 op（如 softmax、gelu），保留线性层输出。TorchTitan 的 policy 函数可以精确控制哪些 op 被 checkpoint。

实践建议：先 profile 每层各子模块的激活大小，对大头做 checkpoint，小头保留。

### 4. AC + FSDP 的交互

关键规则：**AC 必须在 FSDP wrapping 之前应用**。

原因：FSDP 对 module 做 flatten + shard，如果先 wrap FSDP 再加 AC，checkpoint boundary 会打断 FSDP 的 communication hook，导致 all-gather 时序错误。正确顺序：

```python
# 1. 先对 layer 应用 activation checkpoint
for layer in model.layers:
    checkpoint_wrapper(layer, policy=selective_policy)

# 2. 再用 FSDP wrap
model = FSDP(model, ...)
```

TorchTitan 内部也是这个顺序：`apply_ac` → `apply_fsdp`。

### 5. AC + Gradient Accumulation

当使用 gradient accumulation（多个 microbatch 累加梯度）时，AC 的收益更大：

- 没有 AC：每个 microbatch 的激活都要保留到该 microbatch 的 backward 结束，峰值显存 = 单 microbatch 激活 × microbatch 数（如果 backward 延后）。
- 有 AC：每个 microbatch forward 后立即可释放激活，backward 时重算。峰值只需保留 1 个 microbatch 的重算临时激活。

因此 AC 和 gradient accumulation 配合使得可以用更多 microbatch 来模拟大 batch，而不需要线性增加显存。

### 6. 何时不需要 AC

以下场景 AC 收益不大甚至有害：

- **小模型**（如 < 1B 参数）：激活占显存比例低，参数和优化器状态是主要开销，AC 只增加计算不减轻瓶颈。
- **sequence length 很短**：attention score matrix 很小，激活总量有限。
- **已有足够显存**：如果 batch_size=1 且显存充裕，AC 只会拖慢训练。
- **inference 阶段**：不需要 backward，没有激活保存问题。

### 7. 替代方案：Activation Offload

不重算，而是把激活搬到 CPU：

- **优点**：不增加计算量，backward 时从 CPU 拷回即可。
- **缺点**：受 PCIe 带宽限制（~32 GB/s for Gen4 x16），如果激活量大且层数多，offload 延迟可能超过重算时间。
- **适用场景**：单卡训练大模型、PCIe 带宽充裕（如 NVLink 连接的 CPU-GPU）、激活量适中。

实践对比：

| 方法 | 额外计算 | 额外通信 | 显存节省 |
|---|---|---|---|
| Full AC | ~33% | 无 | 最大（O(1) 激活） |
| Selective AC | ~10-20% | 无 | 中等 |
| Offload to CPU | 无 | PCIe 往返 | 最大 |
| 不做 AC | 无 | 无 | 无 |

选择标准：如果 GPU compute 是瓶颈用 offload；如果 PCIe 是瓶颈用 AC；如果都不是瓶颈就不用 AC。
