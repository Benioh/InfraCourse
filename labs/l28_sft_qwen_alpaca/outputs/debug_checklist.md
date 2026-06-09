# L29 Debug Checklist：SFT Loss Mask

## 1. 固定现场

- 记录命令、配置文件、git commit、Python 环境、模型、tokenizer 和数据来源。
- 保存 `config.resolved.yaml`、`metrics.jsonl`、`report.md` 和关键 artifacts。
- 标明这是 patch-test、CPU smoke、Trainer smoke 还是真实训练。

## 2. 数据入口

| 检查项 | 期望 | 异常信号 |
|---|---|---|
| messages 非空 | 至少一条消息 | 直接抛 `ValueError` |
| role 合法 | `system/user/assistant` | unknown role |
| user 不连续 | user 后应接 assistant 或 system | two consecutive user messages |
| 结尾 role | 最后一条为 assistant | 没有监督目标 |

## 3. Tokenizer 与特殊 token

- [ ] `tokenizer.encode(..., add_special_tokens=False)` 是否被使用。
- [ ] `pad_token_id` 是否来自显式参数、tokenizer pad、tokenizer eos 或兜底 0。
- [ ] `eos_token_id` 缺失时是否能退到 pad。
- [ ] 真实 tokenizer 的训练模板和推理模板是否一致。

## 4. Labels 与 Attention Mask

| 位置 | `labels` | `attention_mask` |
|---|---:|---:|
| system/user token | `-100` | `1` |
| assistant token | token id | `1` |
| assistant EOS | eos id | `1` |
| pad | `-100` | `0` |

## 5. Loss 检查

- [ ] `sft_loss` 是否使用 `ignore_index=-100`。
- [ ] 展平 logits 时 vocab 维是否保留为最后一维。
- [ ] 与 `F.cross_entropy(..., reduction="mean")` 是否数值一致。
- [ ] 若 loss 过低，先检查 prompt token 是否泄漏进 labels。
- [ ] 若 loss 为 NaN，先检查 labels 是否全为 `-100` 或 logits 是否含 NaN。

## 6. 结束条件

- 最小复现命令可重复运行。
- assistant token 计数、pad 位置和 EOS 位置都有证据。
- 结论写入 `sft_training_template.md`，并说明 CPU smoke 与真实训练的边界。
