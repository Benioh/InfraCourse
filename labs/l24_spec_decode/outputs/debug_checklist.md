# L25 Spec Decode Debug Checklist

## 1. 先确认实验边界

- 命令、git revision、配置文件、模型、tokenizer、dtype、硬件、并发和随机种子是否记录完整。
- 当前是 patch-test、CPU smoke、MiniInfra bench，还是真实 vLLM/SGLang 服务。
- baseline 是否关闭 spec decode，并使用同一批 prompt、采样参数和并发。

## 2. 低接受率

- 检查 proposer 类型：n-gram、独立 draft model、Medusa/EAGLE 或其他方法。
- 检查 tokenizer/vocab 是否与 target 对齐。
- 检查 prompt domain 是否偏离 draft 训练域或 prompt lookup 的重复模式。
- 检查 temperature、top_p、top_k 是否提高了候选随机性。
- 检查 K 是否过大，早期 mismatch 是否浪费大量候选。
- 分桶查看 per-position acceptance，确认是否只有前一两个位置有收益。

## 3. Draft 成本和显存

- 记录 target 权重、target KV、draft 权重、draft KV、candidate buffer 和临时 logits。
- 对独立 draft model，检查 draft TP、quantization、attention backend 和最大上下文长度。
- 对 n-gram，确认额外 GPU memory 接近 0，但候选覆盖率可能不足。
- 若 OOM，先判断是权重、KV、batch/K 放大项还是临时 buffer 触发。

## 4. KV 和提交长度

- 手算一轮 `accepted_tokens` 与 `num_accepted_drafts`，确认 bonus 没有计入 draft acceptance。
- 接受 m 个 draft 时，有效输出长度应增加 `m + 1`。
- 检查 SGLang `accept_lens` 是否包含 bonus，并确认统计时做了 `-1`。
- 检查 overallocated KV 是否被释放或通过 length/page table 忽略。

## 5. p99 抖动

- 按 accept count、method、batch size、prompt domain 和输出长度分桶 ITL。
- 对比 p50 和 p99，确认收益是否只出现在平均值。
- 查看 accept=0 的比例，早拒绝越多，draft 成本越容易拉高长尾。
- 对混合 workload，考虑按 domain/长度分流、动态 K 或滑动窗口关闭 spec。

## 6. 报告最低字段

- baseline tokens/s、TTFT、ITL p50/p99、GPU memory。
- spec method、K、draft cost、acceptance rate、mean acceptance length、per-position acceptance。
- draft latency、draft memory、accepted tokens/s、drafted tokens/s。
- 输出质量或评测指标，确认候选机制没有改变目标语义。
- 结论要说明适用的 workload、硬件和并发条件。
