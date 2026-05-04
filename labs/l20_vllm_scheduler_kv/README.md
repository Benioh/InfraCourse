# L07.5 · vLLM Scheduler & KV Block Manager：把请求生命周期跑起来

> 本关补上 vLLM 的框架主线：实现 `KVCacheManager` 和 `Scheduler` 的最小同构切片，
> 然后用 `scripts/run_serving_drill.py` 在真实形状的 workload 上把请求驱动一遍——
> 你能第一次"看见" TTFT / ITL / KV pressure 这三层指标。

## 闭环

```bash
cat labs/l20_vllm_scheduler_kv/patch/task.md
$EDITOR labs/l20_vllm_scheduler_kv/patch/starter/scheduler.py
make patch-test M=l20_vllm_scheduler_kv

# 在合成 workload 上演练
bash labs/l20_vllm_scheduler_kv/scripts/run_serving_drill.sh
```

## 测试覆盖（patch 层）

| 测试 | 验证 |
|---|---|
| `test_kv_cache_allocate_free_snapshot` | 分配/释放/快照基本契约 |
| `test_kv_cache_exhaustion_raises` | KV 不足时抛 `RuntimeError` |
| `test_scheduler_moves_waiting_to_running_and_decode` | waiting → running → decode 状态流 |
| `test_scheduler_respects_max_running_and_leaves_waiting` | `max_num_running_reqs` 限制生效 |
| `test_finished_request_frees_kv_blocks` | finished 必须释放 KV |

## Drill 演练（scripts 层）

`scripts/run_serving_drill.py` 在合成 workload（混合长 prompt / 短 prompt / 不同
`max_tokens`）上跑 N 个 schedule 周期，并收集：

- **TTFT**：每个 request 从 add 到首次出现在 `scheduled_decode` 的 step 数
- **ITL**：每个 request 在 decode 阶段平均每 step 产 token 数
- **KV pressure timeline**：每 step 的 `kv_cache_manager.usage`（0.0–1.0）
- **Waiting 队列长度** 时间序列
- **Throughput**：单位 step 完成 request 数

acceptance：
- p50 KV usage > 0.3（说明 workload 真的把缓存压上来了）
- 没有死锁（所有 request 终结）
- 当 KV 不足时 waiting 队列必须增长（不能默默丢请求）

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 默认 drill：8 块 KV，6 个 request |
| `configs/4090_qwen.yaml` | `vllm serve Qwen2.5-0.5B` 的真实启动模板 + bench prompt 集 |
| `configs/h200_qwen2_7b.yaml` | 8×H200 上 7B 服务模板 |

`scripts/run_vllm_serve.sh` 会按 profile 打印 `vllm serve ...` 命令到
`runs/<run-id>/artifacts/vllm_serve_command.sh`，并尽力执行（若 vllm 没装则只生成命令）。

`scripts/bench_ttft.py` 可以把 prompt 集打到一个真实 vLLM 服务，输出 TTFT/ITL CDF。

## 调试工单

见 `tickets/INDEX.md`。建议至少做 `vllm_kv_oom_under_burst` 和 `vllm_waiting_starvation`。

## 进入下一关

`make patch-test` + drill 通过后，进入 [L08 SGLang core](../l21_sglang_serving_core/README.md)。
