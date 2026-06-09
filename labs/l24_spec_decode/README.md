# L25 · Speculative Decoding Greedy Verify

<!-- LECTURE_FIRST_START -->

本讲讲推理服务里的 speculative decoding。目标模型每生成一个 token 都要跑一次 decode step；draft proposer 先提出多个候选 token，target 一次 verify forward 检查连续前缀，接受的 draft 越多，平均每个输出 token 分摊到的 target 成本越低。L25 用一个 CPU-safe 的 greedy verify patch 建立接受前缀、mismatch 接管和 bonus token 的行为合同，再把它接回 MiniInfra 的 acceptance 指标、vLLM speculative 配置和 SGLang KV 边界。

## 学习路线

建议按下面顺序走，先把系统讲通，再写 patch。

1. 读 [system_map.md](system_map.md)：确认 L25 在 Serving 解码加速路径中的位置。
2. 读 [lecture.md](lecture.md)：从 decode step 成本、greedy verify 合同、acceptance 账本讲到生产排查。
3. 读 [source_walkthrough.md](source_walkthrough.md)：按 patch、MiniInfra、vLLM metrics 和 SGLang KV 主路径阅读。
4. 跑 notebook：[n18_spec_decode_acceptance.ipynb](../../notebooks/n18_spec_decode_acceptance.ipynb)，观察 acceptance、draft cost 和 K 的关系。
5. 做 quiz：确认 bonus token、accepted draft count、proposer 家族和 KV 回滚边界。
6. 做 patch：实现最小行为合同并通过测试。
7. 跑 smoke 和 bench：生成 spec decode 指标行，观察并发、p50/p99 和 draft 成本。
8. 填写 [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md)，沉淀一次 spec decode 复盘。

## 本讲定位

| 问题 | 本讲回答 |
|---|---|
| 所属主线 | Serving systems / decode acceleration |
| 核心瓶颈 | target decode step 昂贵、draft 成本、acceptance rate、KV 回滚、并发 p99 |
| 关键机制 | greedy verify、连续前缀接受、mismatch 接管、bonus token、acceptance 与 accepted length 指标 |
| 源码落点 | `patch/reference/spec_decode.py`、`mini_infra/vllm/spec_decode/*`、vLLM `SpeculativeConfig` 与 spec metrics、SGLang accept length/KV 提交 |
| lab 检验 | 全接受、首错、中间 mismatch、k=0、返回类型和 `num_accepted_drafts` 语义 |

## 学完后能做什么

- 解释 target 一次 verify forward 为什么能检查 k 个 draft token 并额外产出一个 bonus token。
- 写出 `greedy_verify(draft_tokens, target_logits)` 的四个边界：全接受、首错、中间错、k=0。
- 区分 `accepted_tokens` 长度和 `num_accepted_drafts`，避免把 bonus token 计入 draft acceptance。
- 用 acceptance rate、draft cost、K、并发和显存判断 spec decode 是否值得打开。
- 看懂 n-gram、独立 draft model、Medusa/EAGLE proposer 在配置、成本和指标上的差别。
- 根据 vLLM 和 SGLang 指标定位低接受率、draft OOM、KV 回滚浪费和 p99 抖动。

## Patch 闭环

```bash
cat labs/l24_spec_decode/patch/task.md
$EDITOR labs/l24_spec_decode/patch/starter/spec_decode.py
make patch-test M=l24_spec_decode
```

smoke：

```bash
python labs/l24_spec_decode/scripts/run_smoke.py --run-id l25_smoke --mode smoke
```

并发指标：

```bash
python labs/l24_spec_decode/scripts/bench_concurrency.py --servers 8001,8002,8003 --batches 1,16,64
```

CPU smoke 只能证明字段、公式和 artifact 路径。真实速度结论必须在目标 GPU、相同模型、相同 prompt 分布、相同采样参数和相同并发条件下，与无 spec baseline 对比。

## 课后产物

| 产物 | 用途 |
|---|---|
| [outputs/debug_checklist.md](outputs/debug_checklist.md) | 排查低接受率、draft OOM、KV 回滚浪费和 p99 抖动 |
| [outputs/source_reading_card.md](outputs/source_reading_card.md) | 复习 patch、MiniInfra、vLLM 和 SGLang spec decode 主路径 |
| [outputs/serving_metrics_template.md](outputs/serving_metrics_template.md) | 记录一次 spec decode serving 的 baseline、acceptance、latency、显存和结论 |

<!-- LECTURE_FIRST_END -->

## 进入下一讲

通过 L25 后进入 L26 评测闭环。下一讲会继续强调同一个原则：任何加速或优化结论都要绑定 workload、baseline、质量指标和可复查 artifact。
