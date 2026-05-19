# L28.5 · Multi-turn Chat Template & Loss Mask

> 多轮 RL / SFT 训练里，一个看似 trivial 的需求会把你折磨到怀疑人生：**给定 messages 列表，
> 返回 (token_ids, loss_mask, attention_mask)，且 assistant 内容被正确识别为 loss target**。
>
> 难点是 chat template 的"条件渲染"：
> - 当 messages 里没有 system 时，模板可能自动插入默认 system；
> - 推理类模型（QwQ-32B / Qwen3）在 assistant 不是最后一条时**会把 `<think>...</think>` 删掉**；
> - 单独 tokenize 每条 message 再拼接，会因为 token 边界融合而和整体 tokenize 结果不一致。
>
> 本关教你 verl PR #1668（Yanbin Jiang）总结出的可靠解法：**Fixed Base Conversation + Delta**。

## 真实事故

参考 [从 tokenizer 视角来分析 Agentic 多轮训练的复杂性](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking_ZH.md)。
verl 团队为这个看似简单的接口重构了两周，跑通了三种朴素方案后才落地"固定 base"。
单独 tokenize 子串 / 用 messages 滑窗做 delta 都各自掉进了不同的坑。

## 闭环

```bash
cat labs/l28.5_multiturn_chat_mask/patch/task.md
$EDITOR labs/l28.5_multiturn_chat_mask/patch/starter/multiturn_tokenizer.py
make patch-test M=l28.5_multiturn_chat_mask
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_returns_three_lists_of_equal_length` | 三个返回值长度一致，attention 全 1 |
| `test_user_tokens_masked_to_neg_100` | user/system/tool 位置 loss = -100 |
| `test_assistant_tokens_are_loss_targets` | assistant 内容 token 在 loss 位置可解码 |
| `test_multi_turn_alignment` | [s,u,a,u,a]：两个 a 都正确进入 loss |
| `test_handles_default_system_injection` | 当 tokenizer 默认补 system 时仍正确 |
| `test_handles_qwq_think_drop` | QwQ 模式（非末尾 assistant 砍 think）下仍保留 think 进 loss |
| `test_tool_message_treated_as_non_assistant` | tool 消息全 -100 |

## 卡住怎么办

1. 跑 `notebooks/n23_chat_template_multiturn.ipynb` 把"chat template 在不同位置渲染不一致"的现象先眼见为实。
2. `make patch-hint` 看 TODO；`make patch-show-solution` 看参考解。

## 写完之后你能做什么

- 解释 verl `BASE_CHATML_FORMAT` / `multiturn_tokenization_and_masking` 的全部分支。
- 在多轮 Agentic SFT/RL 数据 pipeline 里给 tool / function-call 消息打正确 mask。
- 看懂 `apply_chat_template(..., return_assistant_tokens_mask=True)` 为什么对 QwQ/Qwen3 不可用。
- 和 RL 训练 / Rollout / 部署三阶段保持 tokenization 一致性。

## 配套源码研读（可选）

- `github_repo/verl/verl/workers/rollout/schemas.py` — verl 真实多轮 mask 实现
- `github_repo/Awesome-ML-SYS-Tutorial/rlhf/verl/multi-turn/fast_tokenization/` — 整篇必读
