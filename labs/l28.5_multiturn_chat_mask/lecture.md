# L30：Multi-turn Chat Template 与 Loss Mask

## 1. 本讲目标

- 理解多轮 SFT/RL 数据为什么比单轮 SFT 更难 mask。
- 解释 chat template 的条件渲染如何导致 token 边界错位。
- 掌握 Fixed Base Conversation + Delta。
- 正确处理 assistant、system、user、tool 等 role 的 loss。
- 能用 CPU mock tokenizer 验证 default system injection 和 QwQ think drop。

## 2. 问题背景

L29 中，prompt 和 assistant 的边界相对清楚。多轮 agentic 数据会更复杂：一个样本里可能出现 system、user、assistant、tool、assistant 多次交错。训练时仍然只应从 assistant 内容学习，tool observation、user prompt 和 system prompt 都应被 mask 掉。

难点在 chat template。模型自己的模板可能在缺 system 时补默认 system，也可能在某些 role 位置改变 prefix/suffix。推理模型还可能在 assistant 后面有其它消息时删除 `<think>...</think>`。这些条件渲染会让“直接拼字符串”或“滑窗做差”的方案失效。

## 3. 朴素方案的失败点

第一种方案是每条 message 单独套 template、单独 tokenize、再拼接。它会遇到两个问题：缺少上下文时模板可能补默认 system；相邻片段拼接后，BPE 边界可能与整段 encode 不同。

第二种方案是用 `messages[:i]` 和 `messages[:i+1]` 做字符串差。它看起来更接近真实对话，但前缀本身会随着上下文变化。若模板会对非末尾 assistant 删除 think 内容，前后两个字符串就不再是稳定前缀关系。

第三种方案是依赖 `return_assistant_tokens_mask=True`。这个接口依赖具体模板支持 generation 标记，很多模型模板没有覆盖，遇到 QwQ/Qwen3 条件分支时也不可靠。本讲要求自己实现 delta 和 role mask。

## 4. Fixed Base Conversation + Delta

Fixed Base 的核心是先准备一个稳定前缀：

```python
BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]
```

system 占位能阻止默认 system 注入，user 占位让模板进入普通对话路径。先渲染 base 得到 `base_str`。之后每条真实 message 都单独渲染为 `BASE_CONVERSATION + [msg]`，再截掉 `base_str` 得到 `delta_str`。

```text
full_str = template(BASE + [msg])
delta_str = full_str[len(base_str):]
delta_ids = encode(delta_str)
```

每条 message 都在同一个固定上下文里被测量。assistant 在这个局部列表里总是最后一条，因此 QwQ 风格的非末尾 think 删除不会触发。

## 5. Role 到 Loss 的规则

输出是三个等长列表：

| 输出 | 规则 |
|---|---|
| `token_ids` | 累积每条 message 的 `delta_ids` |
| `loss_mask` | assistant 用 `delta_ids`，其它 role 用等长 `-100` |
| `attention_mask` | 本关没有 padding，全部为 `1` |

tool 消息要特别注意。tool 返回是环境输出，不是 assistant 目标。多轮 agentic 训练常有 `assistant -> tool -> assistant`，其中 tool observation 应该进入上下文，但不能进入 loss。

## 6. Mock Tokenizer 覆盖的真实风险

`MockTokenizer(default_system_mode=True)` 模拟缺 system 时自动注入 `DEFAULT_SYS`。Fixed Base 已经带 system，所以真实消息从 user 开始时，输出里不应出现 `DEFAULT_SYS`。

`MockTokenizer(qwq_mode=True)` 模拟非末尾 assistant 删除 `<think>...</think>`。Fixed Base 每次只渲染 `BASE + [msg]`，当前 assistant 是局部最后一条，think 内容应保留在 loss 里。

这些测试说明 patch 不是普通字符串切片。它验证的是模板条件渲染下的稳定边界。

## 7. 测试闭环

`patch/tests/test_patch.py` 覆盖 8 个行为：

1. 三个列表长度一致，attention 全 1。
2. loss 值只能是 `-100` 或同位置 token id。
3. 没有 assistant 时，全是 `-100`。
4. assistant 内容能从 loss 解码。
5. 多轮两个 assistant 都进入 loss。
6. 默认 system 不泄漏。
7. think 内容不被删。
8. tool 内容被 mask，后续 assistant 保留。

实现失败时，按测试名称回到对应分支。长度不一致看 extend；assistant 缺失看 role 判断；default system 泄漏看 base；think 缺失看是否用了真实历史做 delta；tool 进入 loss 看非 assistant 分支。

## 8. 真实系统排查

真实 tokenizer 验证要记录四类证据：tokenizer 名称和版本、chat template 文件或配置、测试 messages、decoded `loss_mask` 非 `-100` 片段。只看 patch-test 不足以证明真实 Qwen/QwQ/Qwen3 模板无问题。

训练、rollout、serving 三阶段必须使用一致的模板语义。若训练保留 think，rollout 删除 think，或者 serving 使用手写 prompt，RL 后续指标会很难解释。

## 9. 小结

L30 的关键在于固定模板上下文。Fixed Base Conversation + Delta 让每条 message 的 token 边界可测，role mask 让 assistant-only loss 可解释。这个能力会直接支撑后续 RL 数据、tool 调用和多轮 rollout。

---

## 补充：多轮 Chat Mask 的深层机制与常见 Bug

### BPE 边界问题详解

为什么不能简单拼接各 message 的独立 tokenization？因为 BPE（Byte Pair Encoding）是上下文相关的：

```python
# 假设 tokenizer 会把 "Hello world" 合并成 [Hello, _world]
# 但如果分开 tokenize：
encode("Hello") = [Hello]
encode(" world") = [_world]
concat = [Hello, _world]  # 看起来对

# 但如果前缀是 "Say Hello"：
encode("Say Hello world") = [Say, _Hello, _world]
# 而 encode("Say Hello") + encode(" world") = [Say, _Hello] + [_world]
# 这次恰好一致，但不总是如此
```

当两个片段的拼接点恰好是 BPE 可以跨越的 merge 边界时，联合 tokenize 和分别 tokenize 的结果不同。Fixed Base + Delta 的方案通过始终在相同前缀下 tokenize 来回避这个问题。

### 多轮 Agentic 数据的实际场景

```json
[
  {"role": "system", "content": "You are a coding assistant with tool access."},
  {"role": "user", "content": "请帮我查看 train.py 的第 50 行"},
  {"role": "assistant", "content": "<think>需要读取文件</think>我来帮你查看。\n<tool_call>read_file(\"train.py\", line=50)</tool_call>"},
  {"role": "tool", "content": "loss = F.cross_entropy(logits, labels)"},
  {"role": "assistant", "content": "第 50 行是交叉熵 loss 计算：`loss = F.cross_entropy(logits, labels)`"}
]
```

Loss mask 规则：
- system: 全部 -100
- user: 全部 -100
- assistant (第一条): token ids（包括 think 内容，因为我们希望模型学会思考）
- tool: 全部 -100（环境输出，不是模型生成的）
- assistant (第二条): token ids

### 常见 Bug 清单

| Bug | 症状 | 根因 |
|---|---|---|
| Default system 泄漏 | Loss 中出现 "You are a helpful assistant" | 模板在缺 system 时自动注入默认 system |
| Think 内容消失 | 推理模型训练后不会思考 | 模板对非末尾 assistant 删除 `<think>` |
| Tool response 进入 loss | 模型学会生成 API 返回值 | tool role 没有被 mask |
| BPE 边界错位 | Decoded loss tokens 出现乱码或多/少字符 | 分段 tokenize 后拼接，BPE 不一致 |
| EOS 缺失 | 推理时模型不会停止 | Assistant 最后没有加 EOS 到 loss |
| Attention 跨样本泄漏 | Packed training 效果差 | Attention mask 没有隔离不同 conversation |

### 从 L28.5 到 L29 (VERL) 的连接

多轮 mask 的正确性直接影响 RL 训练质量：
- 如果 rollout 时模板和训练时不同 → policy 看到的 token 序列不同 → log_prob 不可比
- 如果 tool response 进入了 SFT 的 loss → RL 的 reference policy 学会了生成 tool output → KL 计算出错
- 如果 think 在训练时保留但 rollout 时删除 → advantage 估计偏差
