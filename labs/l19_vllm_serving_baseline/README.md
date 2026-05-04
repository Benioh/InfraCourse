# L07 · vLLM Baseline：Typical_p Sampling

> 本关只做一件事：**实现 Locally Typical Sampling 的 logits filter**——主流推理引擎都需要的非贪婪采样选项。

写完这关你能给 vLLM / SGLang 加任意自定义采样策略。

## 闭环

```bash
cat labs/l19_vllm_serving_baseline/patch/task.md
$EDITOR labs/l19_vllm_serving_baseline/patch/starter/typical_p.py
make patch-test M=l19_vllm_serving_baseline
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_typical_p_one_keeps_all` | typical_p=1.0 时 logits 不变 |
| `test_typical_p_zero_keeps_one` | typical_p≈0 时只剩 1 token |
| `test_filter_value_applied` | 被 filter 位置等于 filter_value |
| `test_kept_tokens_dist_minimal` | 保留 token 的 dist 比丢弃的小 |
| `test_works_with_batch` | (B, V) 每行独立处理 |

## 卡住怎么办

1. 看 `notebooks/n07_kv_cache.ipynb` + `n08_prefill_decode.ipynb` 建立推理直觉。
2. `make patch-hint M=l19_vllm_serving_baseline`。
3. `make patch-show-solution M=l19_vllm_serving_baseline`。

## 进入下一关

`make patch-test` 全绿后，下一关 [L07.5 vLLM scheduler/KV](../l20_vllm_scheduler_kv/README.md) 会补请求调度和 KV block 主线。
