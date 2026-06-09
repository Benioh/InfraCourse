# L28 Debug Tickets

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `pd_router_misroute` | prefix cache 命中后仍路由到不合适的 prefill worker | router 必须结合 cached prefix 和 worker load |
| `pd_kv_transfer_lost` | prefill 完成但 decode 不开始 | 每次 route 都要有显式 KV transfer |
| `pd_complete_double` | 同一 request 重复 complete 后 load 异常 | complete_request 幂等释放 |
| `pd_decode_starvation` | decode worker 满了，prefill 仍持续进流量 | router 要看双侧 load |
| `pd_metrics_skew` | metrics 报告的 prefill tokens 大于 prompt tokens | 不能重复计入 active route |

工单 YAML 在 `InfraCourse/tickets/`。
