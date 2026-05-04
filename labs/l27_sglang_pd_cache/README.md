# L09.5 · SGLang PD Disaggregation：Prefill / Decode 分离 + KV transfer

> 本关补上 SGLang 的框架主线：把 `DisaggregationService` 的 route + KV transfer +
> metrics 写出来，然后用 `scripts/run_pd_drill.py` 驱动 prefill / decode 分离的
> workload，看到 prefix-cache 命中如何降低 prefill 工作量、KV transfer 如何成为
> 新的可观测边界。

## 闭环

```bash
cat labs/l27_sglang_pd_cache/patch/task.md
$EDITOR labs/l27_sglang_pd_cache/patch/starter/disagg_service.py
make patch-test M=l27_sglang_pd_cache

bash labs/l27_sglang_pd_cache/scripts/run_pd_drill.sh
```

## Drill 演练

`scripts/run_pd_drill.py` 在合成 workload 上：

1. 按 `cache_hit_rate` 给 request 注入 `cached_prefix_tokens`
2. 调用 `route_request` 选择 prefill/decode worker
3. 调用 `complete_request` 释放 worker load
4. 收集 metrics：prefill_tokens、cached_prefix_tokens、kv_transfers、worker load 分布
5. 比较 prefix cache off vs on 的 prefill saving

acceptance：
- prefix cache 命中时 prefill_tokens 必须减少 ≥ `min_prefill_saving_ratio`
- 每个 request 必须恰好产生 1 条 KV transfer
- complete 后 worker_loads 必须归零

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 默认 drill：2 prefill + 2 decode worker，20 个 request |
| `configs/h200_qwen.yaml` | 真实 SGLang prefill/decode 分离启动模板（Mooncake-style） |

## 调试工单

见 `tickets/INDEX.md`。建议至少做 `pd_router_misroute` 和 `pd_kv_transfer_lost`。

## 进入下一关

`make patch-test` + drill 通过后，进入 [L10 verl RL baseline](../l29_verl_rl_baseline/README.md)。
