# L20 · vLLM Serving Baseline and Typical-p Sampling

L20 进入推理服务主线：先建立 vLLM/OpenAI-compatible serving baseline，再用一个小 patch 实现 locally typical sampling 的 logits filter，理解采样过滤器在 serving 链路中的位置。

## 学习路线

1. 读 [system_map.md](system_map.md)：确认 OpenAI facade、LLMEngine、Sampler、Scheduler 和 KV manager 的分工。
2. 读 [lecture.md](lecture.md)：从 TTFT/ITL baseline、采样参数、typical-p 机制讲到日志解析和复盘。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra、vLLM sampler 和脚本读源码。
4. 跑 notebook：[n07_kv_cache.ipynb](../../notebooks/n07_kv_cache.ipynb)、[n08_prefill_decode.ipynb](../../notebooks/n08_prefill_decode.ipynb)。
5. 做 quiz：检查 serving baseline、采样过滤、logits mask 和 metrics 边界。
6. 做 patch：实现 `typical_p_filter`。
7. 跑 smoke：生成 serving validation artifact。
8. 填写 [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md)。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | Serving systems |
| baseline 解决什么 | 确认 vLLM server 是否可用，并记录请求、TTFT、吞吐和命令 |
| patch 解决什么 | 对 logits 做 typical-p 过滤，保留 surprisal 接近期望信息量的 token |
| 上游 | OpenAI-compatible request、SamplingParams、logits |
| 下游 | sampler、token selection、metrics、serving report |

## Patch 闭环

```bash
cat labs/l19_vllm_serving_baseline/patch/task.md
$EDITOR labs/l19_vllm_serving_baseline/patch/starter/typical_p.py
IMPL=reference make patch-test M=l19_vllm_serving_baseline
```

smoke：

```bash
python labs/l19_vllm_serving_baseline/scripts/run_vllm_lab.py --run-id l20_smoke
```

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 定位端口、模型加载、采样参数、TTFT/ITL 问题 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 MiniInfra、vLLM sampler 和 patch 主路径 |
| [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md) | 记录 serving baseline 和采样实验 |
