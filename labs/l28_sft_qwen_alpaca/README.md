# L29 · SFT：Chat Template、Loss Mask 与 Causal LM 监督

本讲解决 SFT 数据进入训练前的核心问题：把 `system/user/assistant` 消息渲染成 token 序列，并保证只有 assistant 回答参与 loss。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 L29 在 SFT 到 RL 主线中的位置。
2. 读 [lecture.md](lecture.md)：理解 chat template、labels、attention mask 和 `ignore_index`。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、测试、smoke 和真实 tokenizer 代码阅读。
4. 做 [quiz.yaml](quiz.yaml)：检查 loss mask、EOS、pad 和 smoke 边界。
5. 做 patch：补齐 `tokenize_chat_with_loss_mask` 和 `sft_loss`。
6. 跑 smoke：用 CPU synthetic messages 验证 assistant token 计数和 artifacts。
7. 填 [outputs/sft_training_template.md](outputs/sft_training_template.md)：记录一次可复查的 SFT 数据或训练复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 系统位置 | Serving 章结束后，进入 SFT、偏好优化与 RL 主线 |
| 核心输入 | `messages`、tokenizer、`max_length`、pad/eos 配置 |
| 核心输出 | `input_ids`、`labels`、`attention_mask` |
| 最小验收 | prompt labels 全部为 `-100`，assistant labels 等于对应 token id，pad 不进 loss |
| 后续连接 | L30 会处理更复杂的 multi-turn chat template 和 loss mask |

## Patch 闭环

```bash
cat labs/l28_sft_qwen_alpaca/patch/task.md
$EDITOR labs/l28_sft_qwen_alpaca/patch/starter/sft_pipeline.py
make patch-test M=l28_sft_qwen_alpaca
```

参考实现验证：

```bash
IMPL=reference make patch-test M=l28_sft_qwen_alpaca
```

CPU smoke：

```bash
IMPL=reference python labs/l28_sft_qwen_alpaca/scripts/run_sft.py \
  --config configs/cpu_smoke.yaml \
  --run-id l29_validation
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 SFT 数据、mask、pad/eos 和 loss 异常 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速回忆源码主路径 |
| [outputs/sft_training_template.md](outputs/sft_training_template.md) | 记录 SFT 数据处理或训练复盘 |

## 验收边界

Bronze patch 只验证 tokenization 和 loss 的最小合同。真实 Qwen、Llama 或 Mistral SFT 还要检查 tokenizer 的 chat template、数据截断、packing、验证集、learning rate、checkpoint 和推理模板一致性。
