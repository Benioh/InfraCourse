# L21 · vLLM Scheduler & KV Block Manager：把请求生命周期跑起来

这一讲看推理服务的控制面：请求到达后怎样进入 engine，scheduler 怎样决定谁能 prefill、谁继续 decode、谁完成释放 KV cache，以及 KV block 压力怎样变成 TTFT、ITL 和 waiting queue 的变化。

本讲的 lab 很小，只实现一个教学版 `KVCacheManager` 和 `Scheduler`。讲授重点放在请求生命周期、KV 资源合同、真实 vLLM 源码主路径和生产排查思路上。patch 是最后的出口验收。

## 学习路线

建议按下面顺序走，先把系统讲通，再写 patch。

1. 读 [system_map.md](system_map.md)：先知道 L21 在 serving 主线里的位置。
2. 读 [lecture.md](lecture.md)：完整理解 prefill/decode、KV cache、scheduler 状态机、admission、token budget 和 preemption。
3. 读 [source_walkthrough.md](source_walkthrough.md)：跟着路径读 MiniInfra 和真实 vLLM 源码。
4. 跑 notebook：[n07_kv_cache.ipynb](../../notebooks/n07_kv_cache.ipynb) 和 [n08_prefill_decode.ipynb](../../notebooks/n08_prefill_decode.ipynb)。
5. 做 quiz：确认自己能解释请求生命周期和 KV pressure。
6. 做 patch：实现最小 scheduler / KV manager。
7. 跑 drill：观察 TTFT、ITL、KV usage 和 waiting 队列。
8. 填写 [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md)，沉淀本讲的排查结论。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 它属于哪条主线 | Serving control plane |
| 它解决什么问题 | 有限 KV cache 下，请求如何排队、运行、生成和释放资源 |
| 它连接哪些指标 | TTFT、ITL、KV usage、waiting queue、throughput |
| 它连接哪些源码 | `LLMEngine.add_request`、`LLMEngine.step`、`Scheduler.schedule`、`KVCacheManager.allocate_slots`、`SchedulerOutput` |
| lab 检验什么 | waiting/running/finished 状态迁移和 KV block 分配释放合同 |

## 你会学到什么

- OpenAI-compatible API、LLMEngine、Scheduler、KVCacheManager、Worker 的分工。
- Prefill 和 decode 的资源差异，以及 TTFT / ITL 的来源。
- KV cache 为什么会成为 serving 并发上限。
- KV block 的 owner、usage、free、snapshot 这些基本资源合同。
- Scheduler 的 waiting、running、finished 三张表如何共同维护请求生命周期。
- Admission control 如何同时受 running 上限、KV block 和 token budget 约束。
- 真实 vLLM 如何用 `num_computed_tokens` 追赶 `num_tokens_with_spec`，覆盖 chunked prefill、prefix cache 和 spec decode。
- KV pressure 下如何看 preemption、waiting 增长和 finished 释放。

## Patch 闭环

```bash
cat labs/l20_vllm_scheduler_kv/patch/task.md
$EDITOR labs/l20_vllm_scheduler_kv/patch/starter/scheduler.py
make patch-test M=l20_vllm_scheduler_kv
```

patch 通过后跑一次合成 serving drill：

```bash
bash labs/l20_vllm_scheduler_kv/scripts/run_serving_drill.sh
```

## 测试覆盖

| 测试 | 验证 |
|---|---|
| `test_kv_cache_allocate_free_snapshot` | 分配、释放、快照基本契约 |
| `test_kv_cache_exhaustion_raises` | KV 不足时显式失败 |
| `test_scheduler_moves_waiting_to_running_and_decode` | waiting 到 running，再进入 decode |
| `test_scheduler_respects_max_running_and_leaves_waiting` | `max_num_running_reqs` 限制生效，请求保留在 waiting |
| `test_finished_request_frees_kv_blocks` | finished 请求必须释放 KV |

## Drill 演练

`scripts/run_serving_drill.py` 在合成 workload 上跑多个 schedule step，并记录：

- **TTFT**：request 到达 step 到第一次出现在 `scheduled_decode` 的 step 差。
- **ITL**：request 在 decode 阶段的平均 step/token。
- **KV usage**：每 step 的 `kv_cache_manager.usage`。
- **Waiting 队列长度**：观察 admission 是否被资源卡住。
- **Throughput**：单位 step 完成 request 数。

acceptance：

- p50 KV usage > 0.3，说明 workload 真的给 KV 施压。
- 所有 request 能终结，说明没有死锁。
- 高 KV 压力下 waiting 队列会增长，说明 scheduler 没有默默丢请求或乱共享 block。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 TTFT、ITL、waiting queue、KV pressure 的顺序表 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 快速回忆 MiniInfra 和真实 vLLM 的源码主路径 |
| [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md) | 跑 drill 或真实 benchmark 后填写的指标复盘模板 |

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 默认 drill：8 块 KV，6 个 request |
| `configs/4090_qwen.yaml` | `vllm serve Qwen2.5-0.5B` 的真实启动模板和 bench prompt 集 |
| `configs/h200_qwen2_7b.yaml` | 8xH200 上 7B 服务模板 |

`scripts/run_vllm_serve.sh` 会按 profile 生成 `vllm serve ...` 命令到 `runs/<run-id>/artifacts/vllm_serve_command.sh`，环境里装了 vLLM 时会尝试执行。

`scripts/bench_ttft.py` 可以把 prompt 集打到真实 vLLM 服务，输出 TTFT / ITL CDF。

## 调试工单

见 [tickets/INDEX.md](tickets/INDEX.md)。建议至少做 `vllm_kv_oom_under_burst` 和 `vllm_waiting_starvation`。

## 进入下一关

`make patch-test` 和 drill 都通过后，进入 [L22 SGLang core](../l21_sglang_serving_core/README.md)。
