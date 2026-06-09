# L29 Source Reading Card

## 主路径

| 顺序 | 文件 | 读什么 |
|---|---|---|
| 1 | `patch/starter/sft_pipeline.py` | 学生要补的校验、tokenization 和 loss |
| 2 | `patch/reference/sft_pipeline.py` | pad/eos 兜底、assistant labels、pad mask |
| 3 | `patch/tests/test_patch.py` | mask、EOS、pad、loss 数值合同 |
| 4 | `scripts/run_sft.py` | CPU smoke、Trainer 路径和 artifacts |
| 5 | `github_repo/sglang/python/sglang/lang/chat_template.py` | role prefix/suffix 与 prompt 拼接 |
| 6 | `github_repo/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py` | tokenizer class、路径解析和加载兜底 |

## 读源码时的问题

1. 输入对象是什么，长度由谁决定？
2. 哪一行把 prompt token 从 loss 中排除？
3. EOS 是在哪一步加入 labels 的？
4. pad 后 `input_ids`、`labels`、`attention_mask` 是否仍等长？
5. smoke 写出的 artifact 能证明哪一层合同？

## 记忆点

- `labels=-100` 是 loss mask，不是 attention mask。
- assistant EOS 应进入 loss。
- pad 的 attention mask 为 0，label 仍为 `-100`。
- L29 patch 使用简化 role tag；真实模板复杂度在下一讲继续展开。
