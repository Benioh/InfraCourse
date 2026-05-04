# Debug Tickets — L09.5

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `pd_router_misroute` | prefix cache 命中却被路到没缓存的 prefill | router 必须看 cached_prefix_tokens 选 worker |
| `pd_kv_transfer_lost` | prefill 完成但 decode 不开始 | KV transfer 必须有显式 ack |
| `pd_complete_double` | 同一 request 两次 complete 把 load 减成负数 | complete_request 幂等 |
| `pd_decode_starvation` | decode worker 满了 prefill 还在涌入 | router 要看双侧 load |
| `pd_metrics_skew` | metrics 报告 prefill_tokens > sum(prompt_tokens) | 不能重复计入 |

工单 YAML 在 `InfraCourse/tickets/`。
