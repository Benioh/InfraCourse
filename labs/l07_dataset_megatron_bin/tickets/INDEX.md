# Debug Tickets — L08

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `data_idx_offset_off_by_4` | offsets 单位混淆（字节 vs 元素） | offset 必须以 dtype 字节数对齐 |
| `data_uint16_token_overflow` | 32k+ vocab 用 uint16 写出去再读全是 noise | dtype 边界检查 |
| `data_skipped_blank_silent` | 空 text 行被静默写成 0 length 样本 | 空行必须跳过 |
| `data_mmap_leak_on_reload` | 多次构造 IndexedDataset 之后内存爆 | mmap close / context |
| `data_dedup_then_tokenize_collision` | dedup 后 tokenize 仍有大量重复序列 | raw + token 两层 dedup |

工单 YAML 在 `InfraCourse/tickets/`。
