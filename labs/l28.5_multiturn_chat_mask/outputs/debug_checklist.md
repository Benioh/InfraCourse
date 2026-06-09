# L30 Debug Checklist：Multi-turn Loss Mask

## 1. 固定现场

- 记录 tokenizer 名称、版本、模型路径和 chat template 配置。
- 保存测试 messages、decoded token_ids、decoded loss 片段和 patch-test 输出。
- 标明验证对象是 CPU mock 还是真实 tokenizer。

## 2. Base 检查

| 检查项 | 期望 |
|---|---|
| BASE 包含 system | 防止默认 system 注入 |
| BASE 包含 user | 进入稳定普通对话路径 |
| base_str 固定 | 每条 msg 都从同一个前缀截 delta |
| `BASE + [msg]` | 当前 msg 在局部模板中是最后一条 |

## 3. Delta 检查

- [ ] `full_str` 是否以 `base_str` 为前缀。
- [ ] `delta_str = full_str[len(base_str):]` 是否为空异常。
- [ ] `encode(delta_str, add_special_tokens=False)` 是否被使用。
- [ ] token_ids、loss_mask、attention_mask 是否等长。

## 4. Role Mask 检查

| Role | loss_mask |
|---|---|
| assistant | 同位置 token id |
| system | `-100` |
| user | `-100` |
| tool / function | `-100` |

## 5. 条件渲染检查

- [ ] default system mode 下，输出中不应出现 `DEFAULT_SYS`。
- [ ] QwQ mode 下，assistant 的 `<think>...</think>` 应保留在 loss 中。
- [ ] 多轮两个 assistant 都应在 decoded loss 里出现。
- [ ] user 问题和 tool observation 不应出现在 decoded loss 中。

## 6. 结束条件

- patch-test 通过。
- decoded loss 片段只包含预期 assistant 内容。
- 真实 tokenizer 验证已写入 `multiturn_mask_template.md`。
