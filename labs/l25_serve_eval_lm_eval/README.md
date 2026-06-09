# L26 · Serving Eval：OpenAI-Compatible GSM8K Harness

<!-- LECTURE_FIRST_START -->

本讲讲推理服务上线后的评测闭环。Serving 优化如果只报告吞吐和延迟，很容易把质量回退、stop 设置错误、timeout 漏题或 prompt 格式偏移藏起来。L26 用一个 CPU-safe 的 GSM8K-style harness 建立从样本、few-shot prompt、OpenAI-compatible endpoint、双指标打分到 artifact 落盘的最小合同，再对照 SGLang eval 和 vLLM/SGLang OpenAI endpoint。

## 学习路线

建议按下面顺序走，先把评测链路讲通，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L26 在 Serving 质量闭环中的位置。
2. 读 [lecture.md](lecture.md)：从为什么需要质量评测、prompt/stop/temperature 边界讲到 artifact 复盘。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、run_eval、SGLang eval 和 OpenAI endpoint 主路径阅读。
4. 做 quiz：确认 exact match、first-number match、few-shot leakage、stop sequence 和 completion endpoint 的边界。
5. 做 patch：实现最小 evaluation harness 并通过测试。
6. 跑本地 stub eval：不依赖真实服务，生成 `metrics.jsonl`、`eval_summary.json` 和 `report.md`。
7. 如有本地 vLLM/SGLang 服务，再切到 OpenAI-compatible 配置重跑。
8. 填写 [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md)，沉淀一次评测报告。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | Serving systems / quality evaluation |
| 核心风险 | 只看性能不看质量、prompt leakage、stop 缺失、temperature 漂移、timeout 被当成错误答案 |
| 关键机制 | few-shot prompt、completion client、exact match、first-number match、per-sample artifact、completion rate |
| 源码落点 | `patch/reference/eval_harness.py`、`scripts/run_eval.py`、SGLang simple eval、vLLM/SGLang OpenAI completion endpoint |
| lab 检验 | 数字抽取、字符串 exact match、数字 match、few-shot 顺序、stub client 完整评测和部分正确聚合 |

## 学完后能做什么

- 写出最小评测 harness：切分 shots/eval items、构造 prompt、调用 `client.complete()`、聚合指标。
- 区分 `exact_match` 与 `first_number_match` 各自能解释的错误类型。
- 解释为什么 deterministic eval 通常要固定 temperature、stop sequence、max tokens 和 endpoint。
- 判断一份 serving 评测报告是否包含样本级记录、完成率、质量指标、配置快照和命令。
- 对照 SGLang eval 与 vLLM/SGLang OpenAI endpoint，定位真实服务多出的 timeout、并发、token usage 和路由复杂度。

## Patch 闭环

```bash
cat labs/l25_serve_eval_lm_eval/patch/task.md
$EDITOR labs/l25_serve_eval_lm_eval/patch/starter/eval_harness.py
make patch-test M=l25_serve_eval_lm_eval
```

本地 stub eval：

```bash
python labs/l25_serve_eval_lm_eval/scripts/run_eval.py \
  --config configs/local_stub.yaml \
  --run-id l26_stub
```

脚本包装：

```bash
RUN_ID=l26_stub bash labs/l25_serve_eval_lm_eval/scripts/run_eval.sh
```

真实服务评测需要先启动 OpenAI-compatible endpoint，再使用 `configs/local_vllm_qwen.yaml` 或同结构配置。stub 只证明 harness、artifact 和指标字段；真实质量结论要绑定模型、服务端版本、prompt 集、采样参数和完成率。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查 exact 为 0、数字指标偏高、stop 缺失、timeout 和 endpoint 配置错误 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、run_eval、SGLang eval 和 OpenAI endpoint 主路径 |
| [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md) | 记录一次 serving eval 的配置、样本、质量指标、完成率和后续动作 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L26 后进入 L27 SGLang PD Observability。下一讲会把这里的 artifact 习惯继续用于服务观测：指标必须能追到命令、配置、输入和源码状态。
