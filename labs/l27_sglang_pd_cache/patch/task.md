# L28 Patch · SGLang-shaped PD Disaggregation Service

## 你要交付什么

实现一个最小 SGLang-shaped `DisaggregationService`。它负责把一个请求路由到 prefill/decode worker，按 prefix cache 命中计算 prefill 工作量，记录 prefill 到 decode 的 KV transfer，并暴露可用于 debug 的 metrics。

```python
class DisaggregationService:
    def route_request(self, request_id, prompt_token_count, cached_prefix_tokens=0) -> DisaggRoute: ...
    def transfer_kv(self, request_id, prefill_worker, decode_worker, token_count) -> KVTransfer: ...
    def metrics(self) -> dict: ...
    def transfers_for_request(self, request_id) -> list[KVTransfer]: ...
    def complete_request(self, request_id) -> None: ...
```

补丁规模目标：80 到 130 行。

## 数据模型

- `Worker`：`worker_id`、`role`、`load_tokens`、`active_requests`。
- `KVTransfer`：一次 prefill 到 decode 的 KV handoff。
- `DisaggRoute`：一次活跃 route 的请求、worker、prompt/cache/prefill/transfer token 记录。
- `self.routes`：当前活跃请求。
- `self.transfers`：历史 transfer 记录。

## 不变量

1. `route_request` 必须校验 request id、prompt token 数和 cached prefix token 数。
2. 同一个活跃 request 不能重复 route。
3. `prefill_tokens = prompt_token_count - cached_prefix_tokens`。
4. Prefix cache 只减少 prefill 工作量；本关 `kv_transfer_tokens = prompt_token_count`。
5. Prefill/decode worker 都按当前 `load_tokens`、`active_requests`、`worker_id` 选择 least-loaded。
6. `route_request` 必须记录一个 `KVTransfer`。
7. `metrics()` 必须返回 transfer 数、总 transfer token 数、prefill token 数、cached prefix token 数、active requests、按 worker 聚合的 token 数和当前 worker loads。
8. `transfers_for_request()` 必须保持插入顺序。
9. `complete_request()` 必须释放 active worker load；unknown request 是 no-op。
10. complete 不应清空历史 transfer。

## 怎么验证

```bash
make patch-test M=l27_sglang_pd_cache
```

7 个测试：

| 测试 | 验证 |
|---|---|
| `test_route_request_records_worker_choice_and_kv_transfer` | route 字段、transfer 和 worker load |
| `test_route_request_selects_least_loaded_workers` | 多 worker 下选择 least-loaded |
| `test_route_request_validates_required_fields` | 输入校验 |
| `test_metrics_aggregate_cache_prefill_and_decode_worker_tokens` | metrics 聚合 |
| `test_transfers_for_request_filters_in_order` | transfer 历史过滤与顺序 |
| `test_complete_request_releases_active_worker_load` | complete 释放与 no-op |
| `test_empty_metrics_are_zero` | 空状态 shape |

Drill：

```bash
IMPL=reference python labs/l27_sglang_pd_cache/scripts/run_pd_drill.py \
  --config configs/cpu_smoke.yaml \
  --run-id l28_pd
```

## 写完之后你能做什么

- 解释 SGLang PD 分离里 route、prefix cache 和 KV handoff 的边界。
- 判断 cache on/off 是否真的降低 prefill 工作量。
- 用 metrics 定位 transfer lost、decode starvation、complete double 和 worker load leak。
- 把课堂整数模型映射到真实 SGLang RadixCache、disagg queues 和 bootstrap service。
