# L29：SFT Loss Mask 与 Chat Tokenization

## 1. 本讲目标

- 说清 SFT 在 RLHF/RL 主线中的位置。
- 理解 `messages -> input_ids/labels/attention_mask` 的数据合同。
- 解释为什么 prompt token 要写成 `-100`，assistant token 才是学习目标。
- 掌握 EOS、pad、截断和 `ignore_index=-100` 的边界。
- 能用 patch tests 和 CPU smoke 验证一次 SFT 数据处理链路。

## 2. 问题背景

SFT 的任务很具体：给定一段 chat 数据，让 causal LM 在 prompt 条件下学习 assistant 回答。训练样本通常来自 `system/user/assistant` 消息，而模型吃到的是 token id。中间需要一个可靠的数据转换层，把消息转成 `input_ids`，同时构造同长度的 `labels` 和 `attention_mask`。

这里最容易出错的是 loss mask。若 prompt token 也进入 loss，模型会被训练去复述 system 或 user 内容；若 assistant token 被 mask 掉，样本看起来存在，监督信号却消失；若 pad 进入 loss，训练会学习无意义的补齐 token。L29 的 patch 就是把这些底层合同固定下来。

## 3. SFT 在系统里的位置

在完整 RLHF 流程中，SFT 位于偏好优化和 RL 之前。它先把 base model 调到能按指令回答的状态，再交给 DPO、PPO、GRPO 或 reward-driven 训练继续对齐。

```text
pretrained LM
  -> SFT data processing
  -> SFT training
  -> instruction-following checkpoint
  -> preference optimization / RL
```

SFT 的工程质量会影响后面的所有阶段。模板不一致会让训练和推理看到不同格式；loss mask 错位会污染监督目标；EOS 缺失会让模型推理时更难停下。

## 4. 三个同长度序列

`tokenize_chat_with_loss_mask` 返回三个列表：

| 字段 | 含义 |
|---|---|
| `input_ids` | 模型实际输入的 token id |
| `labels` | 训练目标；prompt 和 pad 为 `-100`，assistant 为对应 token id |
| `attention_mask` | 真实 token 为 `1`，pad 为 `0` |

这三个列表必须等长。位置 `i` 的三项描述同一个 token：它的输入 id、它是否要计入 loss、它是否是 pad。测试会检查 pad 末尾的 label 是否为 `-100`，也会检查 assistant token 是否被保留下来。

## 5. Chat Template 的简化合同

真实 Qwen、Llama、Mistral tokenizer 通常通过 chat template 把 messages 渲染成带角色标记的字符串。L29 的 patch 用更小的合同训练这个概念：每条消息编码为 `<|role|>{content}`，并使用 `tokenizer.encode(..., add_special_tokens=False)` 得到 segment token。

简化合同的好处是可测。`_CharTokenizer` 把每个字符稳定映射成一个 token id，测试就能精确判断哪些位置属于 user，哪些位置属于 assistant。真实 tokenizer 会有 BPE 合并、特殊 token 和模板条件分支，这些复杂度放到 L30 继续处理。

## 6. Labels 如何写

SFT 的 labels 规则很短：

```text
system/user segment       -> labels = -100
assistant segment         -> labels = input_ids
assistant trailing EOS    -> labels = eos_id
pad segment               -> labels = -100
```

`-100` 是 PyTorch `cross_entropy` 常用的 ignore index。对应位置不计算 loss，也不进入平均分母。这个设计让 prompt 仍然作为条件输入被模型读取，但不会要求模型学习“生成 prompt”。

EOS 要放在 assistant loss 里。模型在训练时看见回答结束符，推理时才更容易学会停止。若 pad token id 与 EOS token id 相同，也要通过 `attention_mask=0` 和 `labels=-100` 把 pad 位置排除。

## 7. `sft_loss` 的边界

本讲的 `sft_loss` 假设 logits 和 labels 已经按同一位置对齐，不在函数里做 causal shift。它只负责把 `[B, T, V]` logits 展平成 `[B*T, V]`，把 `[B, T]` labels 展平成 `[B*T]`，再调用：

```python
F.cross_entropy(
    logits.reshape(-1, vocab),
    labels.reshape(-1),
    ignore_index=-100,
    reduction="mean",
)
```

真实训练框架经常在 model forward 内部或 collator 之后处理 shift。读代码时先确认当前接口期待的是 pre-shifted labels，还是由模型内部完成 next-token 对齐。

## 8. Smoke 路径

`scripts/run_sft.py` 有两条路径。CPU `tokenize_only` 模式用 synthetic messages 和 `_CharTokenizer` 跑 patch 函数，写出 `tokenize_summary.json` 与 `metrics.jsonl`。训练模式会加载 HuggingFace tokenizer、模型和数据集，再把 tokenized dataset 交给 `Trainer`。

本地验证优先跑 CPU smoke，因为它能快速确认 mask 和 artifacts：

```bash
IMPL=reference python labs/l28_sft_qwen_alpaca/scripts/run_sft.py \
  --config configs/cpu_smoke.yaml \
  --run-id l29_validation
```

真实训练结果要额外记录模型名、数据量、max length、learning rate、batch、gradient accumulation、dtype 和硬件。没有这些条件，loss 曲线不能和其他 run 比较。

## 9. 排障顺序

1. 先看 role order：messages 是否为空、role 是否合法、最后一条是否 assistant。
2. 再看 tokenizer：pad/eos 是否存在，`encode` 是否使用了预期参数。
3. 再看 labels：prompt 是否全为 `-100`，assistant 是否保留 token id，EOS 是否进入 loss。
4. 再看 pad：pad 后 `attention_mask=0`，labels 仍为 `-100`。
5. 最后看 loss：`sft_loss` 是否与 PyTorch CE 数值一致。

这条顺序能把数据错误和训练错误分开。很多 SFT 问题看起来像 learning rate 或模型能力问题，根因却是 mask、EOS 或模板不一致。

## 10. 小结

L29 不追求完整 SFT 框架，而是固定最小数据合同。只要学生能解释每个 token 的输入、label 和 mask，就能继续理解 L30 的多轮模板、L31 之后的 RL 数据流，以及真实 Trainer 中的 collator、padding 和 loss 计算。

---

## 补充：SFT 工程深入

### Causal Shift（Next-Token Prediction 对齐）

SFT 使用 next-token prediction 目标。这意味着位置 `t` 的 logits 预测的是位置 `t+1` 的 token：

```
Position:    0    1    2    3    4
Input:      [A]  [B]  [C]  [D]  [E]
Logits:     预测B 预测C 预测D 预测E 预测?
Labels:     [B]  [C]  [D]  [E]  [-100]
```

**两种处理方式**：
1. **Model 内部 shift**：HuggingFace 的 `CausalLMOutputWithPast` 在 `loss` 计算时自动做 `logits[..., :-1, :]` vs `labels[..., 1:]`。
2. **外部 pre-shift**：在 collator 中就把 labels 左移一位。此时 model 内部不再 shift。

**危险**：如果你的代码同时做了两次 shift（collator shift 了一次，model 又 shift 了一次），labels 就错位了两个位置，训练会学到错误的目标。

### Batch Collation 与 Padding

真实训练中一个 batch 内的样本长度不同，需要 padding：

```python
def sft_collator(features, tokenizer, max_length):
    # 1. 截断超长样本
    # 2. Pad 短样本到 batch 内最长（或 max_length）
    # 3. Padded 位置: input_ids=pad_id, labels=-100, attention_mask=0
```

**Left-padding vs Right-padding**：
- 训练时通常 right-pad（pad 在序列末尾）
- 推理时 left-pad（pad 在序列开头，让最后一个 token 始终在最右边方便生成）
- 不一致会导致 position_ids 错误

### 常见 SFT 失败模式

| 现象 | 可能原因 |
|---|---|
| Loss 很低但生成质量差 | Prompt 进入了 loss（模型在背诵 prompt） |
| Loss 不降 | Labels 全是 -100（没有监督信号） |
| 生成时不停止 | EOS 没有进入 loss / pad_token == eos_token 冲突 |
| 生成重复内容 | 训练数据中有大量重复样本 |
| 格式不对（缺 role 标记） | 推理时的 chat template 和训练时不一致 |
| 多卡训练 loss 和单卡不同 | Loss reduction 方式不同（per-token mean vs per-sample mean） |

### Loss Reduction 的陷阱

```python
# 方式 1: Per-token mean（PyTorch 默认）
loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction="mean")
# 分母 = 所有非 -100 的 token 总数

# 方式 2: Per-sample mean
per_sample_loss = per_token_loss.sum(dim=-1) / mask.sum(dim=-1)
loss = per_sample_loss.mean()
# 分母 = batch_size（每个样本权重相等，不管长短）
```

方式 1 让长样本的权重更大（贡献更多 token）。方式 2 让每个样本权重相等。选错会导致"长样本被过度训练"或"短样本被忽略"。

多卡 DDP 下还要注意：每个 rank 的 batch 中非 -100 token 数量不同，如果 loss 已经 mean 了再 all-reduce 取平均，等价于按 rank 而不是按 token 做平均。正确做法是 all-reduce 分子和分母，再相除。
