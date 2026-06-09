# Source Reading Card：L23 SDPA / FlashAttention benchmark

## 主路径

1. `labs/l22_flash_attn_v2_bench/patch/starter/flash_bench.py`
   - 看 `eager_attention()`、`flash_attention()`、`bench_attention()` 的 TODO。
   - 结论：学生要补齐公式、SDPA 调用和指标合同。

2. `labs/l22_flash_attn_v2_bench/patch/reference/flash_bench.py`
   - 看 eager baseline、SDPA wrapper、CUDA synchronize、正确性和性能分层。
   - 结论：fp32 小 shape 用于数值，目标 dtype 大 shape 用于性能。

3. `labs/l22_flash_attn_v2_bench/patch/tests/test_patch.py`
   - 看 shape、causal mask、数值一致、指标字段和 CUDA 条件测试。
   - 结论：测试覆盖最小合同，不覆盖所有生产 backend 组合。

4. `labs/l22_flash_attn_v2_bench/scripts/run_bench.py`
   - 看 `IMPL` 选择、config 解析、逐 shape 运行、metrics/json/csv 输出。
   - 结论：benchmark 要留下可复查 artifact。

5. `mini_infra/gpu/triton_softmax.py`
   - 看 `stable_softmax()`、`online_softmax()`、`max_abs_error()`。
   - 结论：online softmax 维护 running max 和 running sum，解释分块 softmax 的数值基础。

6. `github_repo/torchtitan/torchtitan/models/common/attention.py`
   - 看 layout transpose、`sdpa_kernel()` 和 SDPA 调用。
   - 结论：真实训练代码会控制 backend，并处理 `[B,T,H,D]` 与 `[B,H,T,D]` 的转换。

7. `github_repo/sglang/python/sglang/srt/layers/attention/torch_native_backend.py`
   - 看 KV cache、request token 映射、dtype 对齐和 SDPA 调用。
   - 结论：serving decode 的 SDPA 输入来自 cache 和请求元数据。

8. `github_repo/vllm/vllm/v1/attention/backends/cpu_attn.py`
   - 看 mask、dropout、scale、causal 和 GQA 参数如何传入 SDPA。
   - 结论：真实后端比 patch 多出 mask 和 attention type 的组合。

## 快速自检

- 我能否指出 `[B,H,T,T]` 中间矩阵在哪个 baseline 中产生？
- 我能否解释 SDPA API 和实际 backend 的区别？
- 我能否说明为什么 CPU smoke 不能证明 CUDA speedup？
- 我能否从 `bench_summary.json` 还原每个 shape 的验收结果？
- 我能否列出真实 wrapper 比 patch 多出的四类生产复杂度？
