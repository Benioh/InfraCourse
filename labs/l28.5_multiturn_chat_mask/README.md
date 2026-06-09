# L30 · Multi-turn Chat Template 与 Loss Mask

本讲解决多轮 SFT/RL 数据处理中的 mask 对齐问题：给定一串 `messages`，使用模型自己的 chat template 得到 token 序列，并只让 assistant 内容进入 loss。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L30 接在 L29 SFT loss mask 之后，扩展到多轮模板条件渲染。
2. 读 [lecture.md](lecture.md)：理解 Fixed Base Conversation + Delta。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 starter、reference、mock tokenizer 和 tests 阅读。
4. 做 [quiz.yaml](quiz.yaml)：检查默认 system、think drop、tool role 和 mask 不变量。
5. 做 patch：实现 `tokenize_with_loss_mask`。
6. 跑 patch-test：用 CPU mock tokenizer 验证 8 个边界。
7. 填 [outputs/multiturn_mask_template.md](outputs/multiturn_mask_template.md)：记录真实 tokenizer 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 前置 | L29 已讲单轮 SFT 的 prompt/assistant loss mask |
| 新增复杂度 | 多轮 messages 会触发模板条件渲染和 BPE 边界变化 |
| 核心算法 | Fixed Base Conversation + Delta |
| 最小输出 | `token_ids`、`loss_mask`、`attention_mask` 三个等长列表 |
| 关键边界 | default system injection、QwQ think drop、tool 消息 mask |

## Patch 闭环

```bash
cat labs/l28.5_multiturn_chat_mask/patch/task.md
$EDITOR labs/l28.5_multiturn_chat_mask/patch/starter/multiturn_tokenizer.py
make patch-test M=l28.5_multiturn_chat_mask
```

参考实现：

```bash
IMPL=reference make patch-test M=l28.5_multiturn_chat_mask
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查多轮 loss mask、默认 system、think drop 和 tool 消息 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速回忆源码主路径 |
| [outputs/multiturn_mask_template.md](outputs/multiturn_mask_template.md) | 记录真实 tokenizer 端到端验证 |

## 验收边界

本讲 patch 使用 `_mock_tokenizer.py` 模拟两类真实模板行为：缺 system 时注入默认 system，非末尾 assistant 删除 `<think>...</think>`。Patch-test 证明算法对这些条件渲染保持稳定；真实 Qwen/QwQ/Qwen3 tokenizer 还要单独记录模型路径、tokenizer 版本、template 文件和 decoded loss 片段。
