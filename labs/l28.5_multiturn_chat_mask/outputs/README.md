# L30 课后产物

本目录保存多轮 chat template 与 loss mask 排查材料。

| 文件 | 用法 |
|---|---|
| `debug_checklist.md` | 按 base、delta、role、default system 和 think drop 排查 |
| `source_reading_card.md` | 快速回忆源码主路径 |
| `multiturn_mask_template.md` | 记录真实 tokenizer 验证 |

每次验证真实 tokenizer 时，记录 tokenizer 名称、版本、chat template 文件、messages、decoded token_ids 和 decoded loss 片段。
