# Debug Tickets — L07.5

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `vllm_kv_oom_under_burst` | 突发请求让 KV 占满，新请求 hang | KV pressure 告警、admission 控制 |
| `vllm_waiting_starvation` | 长 prompt 总是被短请求挤掉 | FIFO vs priority、prefill 配额 |
| `vllm_decode_drop` | running 请求偶发不出现在 decode 列表 | running/waiting 状态机维护 |
| `vllm_kv_double_alloc` | 同一 request 被分配两段 KV | finish/free 顺序 |
| `vllm_oom_silent_finish` | KV 不够却显示 finish | RuntimeError 必须冒泡 |

工单 YAML 在 `InfraCourse/tickets/`。
