# L09.5 Patch · SGLang-shaped PD Disaggregation Service

## 你要交付什么

实现 `mini_infra/sglang/srt/managers/disagg_service.py` 的增强版。这个组件要完成一个最小 SGLang PD 闭环：请求进入后选择 prefill/decode worker，按 prefix cache 命中计算 prefill 工作量，记录 prefill → decode 的 KV transfer，并暴露 metrics。

```python
class DisaggregationService:
    def route_request(self, request_id, prompt_token_count, cached_prefix_tokens=0) -> DisaggRoute: ...
    def transfer_kv(self, request_id, prefill_worker, decode_worker, token_count) -> KVTransfer: ...
    def metrics(self) -> dict: ...
    def transfers_for_request(self, request_id) -> list[KVTransfer]: ...
    def complete_request(self, request_id) -> None: ...
```

补丁规模目标：80-130 行。

## 不变量

1. `route_request` 必须校验 request_id、prompt token 数和 cached prefix token 数。
2. `prefill_tokens = prompt_token_count - cached_prefix_tokens`，prefix cache 只减少 prefill 工作量。
3. prefill/decode worker 都按当前 `load_tokens` 最小优先选择。
4. `route_request` 必须记录一个 `KVTransfer`，transfer token 数等于 prompt token 数。
5. `complete_request` 必须释放该请求占用的 worker load，重复 complete 不报错。
6. `metrics()` 必须返回 transfer 数、总 token 数、prefill token 数、cached prefix token 数、按 worker 聚合的 token 数和当前 worker load。
7. `transfers_for_request()` 必须保持插入顺序。

## 怎么验证

```bash
make patch-test M=l27_sglang_pd_cache
```

## 写完之后你能做什么

- 解释 SGLang PD 分离里 KV transfer 是什么系统边界。
- 解释 prefix cache 命中为什么降低 prefill 计算，但不自动消除 decode 侧 KV 交接成本。
- 解释 TTFT/ITL debug 为什么要拆 prefill load、decode load、KV transfer。
- 给 PD benchmark 设计最小 metrics 集合。
