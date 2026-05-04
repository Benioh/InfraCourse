# Debug Tickets — L08.8

| Ticket | 故障形态 | 主要练什么 |
|---|---|---|
| `eval_stop_missing_continues_self_qa` | 模型继续生成下一题答案被算成主答案 | stop=["Question:"] |
| `eval_temperature_drift` | 跑两次结果不同 | temperature=0 / seed |
| `eval_format_only_metric` | exact_match=0 但 first_number_match=高 | 双 metric 才能定位 |
| `eval_few_shot_recency_bias` | 改变 shot 顺序得分波动 5%+ | few-shot ordering |
| `eval_endpoint_timeout_silent` | 一些请求 timeout 被算成空答 | 必须区分 timeout / 空答 |

工单 YAML 在 `InfraCourse/tickets/`。
