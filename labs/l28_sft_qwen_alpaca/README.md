# L09.8 · SFT：用 HuggingFace Trainer 在 Qwen2.5-0.5B 上微调 Alpaca

> 本关只做一件事：**实现 chat-template 下的 prompt-loss-masked SFT** —
> 即只在 assistant 回答 token 上算 loss，prompt token 用 -100 屏蔽。
> 然后用 `scripts/run_sft.py` 在 5k Alpaca 样本上微调 Qwen2.5-0.5B。

工业 RLHF 的第一步永远是 SFT。L10 / L10.5 / L11 直接跳到 PPO，跳过了"先把模型
训成会说话的状态"这一段。L09.8 把这个洞补上。

## 闭环

```bash
cat labs/l28_sft_qwen_alpaca/patch/task.md
$EDITOR labs/l28_sft_qwen_alpaca/patch/starter/sft_pipeline.py
make patch-test M=l28_sft_qwen_alpaca

# 在 4090 / H200 上真实跑 SFT（需要 transformers + datasets）
bash labs/l28_sft_qwen_alpaca/scripts/run_sft.sh
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_chat_template_concat_order` | system → user → assistant 顺序 |
| `test_loss_mask_blocks_prompt_tokens` | prompt 部分 labels == -100 |
| `test_loss_mask_keeps_assistant_tokens` | assistant 部分 labels == input_ids |
| `test_sft_loss_ignores_minus_100` | -100 不进入 cross-entropy |
| `test_sft_loss_matches_torch_ce` | 和 `F.cross_entropy(ignore_index=-100)` 数值一致 |
| `test_pad_token_handled` | pad 也被 mask 成 -100 |

## SFT smoke

`scripts/run_sft.py` 用 HuggingFace `Trainer`：

1. 加载 Qwen2.5-0.5B-Instruct + tokenizer
2. 加载 `tatsu-lab/alpaca` 5k 子集（或本地 jsonl 备份）
3. 用 patch 的 `tokenize_chat_with_loss_mask` 构造 dataset
4. `Trainer(...).train()` 100 步
5. acceptance：loss 单调下降，末段 loss < 1.0 × 起始 loss

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | tokenizer + masking only（不真训） |
| `configs/4090_qwen.yaml` | Qwen2.5-0.5B + Alpaca 5k + 100 步 |
| `configs/h200_qwen2_7b.yaml` | Qwen2.5-7B + Alpaca 50k + 1k 步 |

## 调试工单

见 `tickets/INDEX.md`。建议至少做 `sft_eos_truncation` 与 `sft_loss_dropping_too_fast`。

## 进入下一关

通过后进入 [L10 verl RL baseline](../l29_verl_rl_baseline/README.md)（KL controller）→
[L10.3 DPO loss](../l30_dpo_loss/README.md)。
