# L30 Multi-turn Mask 复盘模板

## 验证对象

- 日期：
- tokenizer / 模型：
- tokenizer 版本：
- chat template 来源：
- 命令：
- git commit：

## Messages

```json
[
  {"role": "system", "content": ""},
  {"role": "user", "content": ""},
  {"role": "assistant", "content": ""},
  {"role": "tool", "content": ""},
  {"role": "assistant", "content": ""}
]
```

## 输出检查

| 项目 | 结果 |
|---|---|
| len(token_ids) |  |
| len(loss_mask) |  |
| len(attention_mask) |  |
| decoded token_ids |  |
| decoded loss tokens |  |
| DEFAULT_SYS 是否泄漏 |  |
| think 是否保留 |  |
| tool 是否 mask |  |

## 结论

- 本次能证明什么：
- 本次不能证明什么：
- 下一步动作：
