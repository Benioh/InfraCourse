# L08.3 · FlashAttention v2 真实库：装一遍 + 跑 benchmark

> 本关只做一件事：**用 `torch.nn.functional.scaled_dot_product_attention`（FA backend）
> 替换 eager attention，并实测显存与吞吐**。如果环境装了 `flash-attn` 包，
> 同时调一遍它的 `flash_attn_func` 验证一致性。

之前 L04.5 学了 ring attention 的*数学*（在线 softmax），但从未碰过真实 FA 库。
工业代码每天都在用 FA2/FA3。L08.3 把这个空白补上。

## 闭环

```bash
cat labs/l22_flash_attn_v2_bench/patch/task.md
$EDITOR labs/l22_flash_attn_v2_bench/patch/starter/flash_bench.py
make patch-test M=l22_flash_attn_v2_bench

# GPU 真实 benchmark
RUN_GPU_TESTS=1 make patch-test M=l22_flash_attn_v2_bench
bash labs/l22_flash_attn_v2_bench/scripts/run_bench.sh
```

## 测试覆盖

| 测试 | 标记 | 验证 |
|---|---|---|
| `test_eager_shape_correct` | cpu | [B, H, T, D] in / [B, H, T, D] out |
| `test_eager_causal_mask_no_leak` | cpu | causal 输出仅依赖 j ≤ i 的 token |
| `test_flash_matches_eager_numerically` | cpu | SDPA 输出与 eager 数值一致（fp32） |
| `test_bench_returns_required_metrics` | cpu | dict 含 eager_time_ms/flash_time_ms/speedup/max_abs_diff |
| `test_bench_speedup_gpu` | gpu | seq=2048 时 speedup ≥ 2.0 |
| `test_bench_peak_memory_gpu` | gpu | flash peak mem ≤ 0.5 × eager peak mem |

## Drill

`scripts/run_bench.py` 在 `(seq_len, head_dim)` 网格上跑：

- `(512, 64)`、`(1024, 64)`、`(2048, 64)`、`(4096, 64)`
- 输出 `runs/.../artifacts/bench.csv`：`seq_len, head_dim, eager_ms, flash_ms, speedup, eager_mem_mb, flash_mem_mb`
- 验证 `flash_attn` 包安装：若已装，多跑一组 `flash_attn_func` 对比

## Configs

| 配置 | 用途 |
|---|---|
| `configs/cpu_smoke.yaml` | 用 `disable_flash` 走 math backend，验证 SDPA 数值正确 |
| `configs/4090_bench.yaml` | seq ∈ {512, 1024, 2048}，断言 speedup ≥ 2× |
| `configs/h200_bench.yaml` | seq ∈ {2048, 4096, 8192}，head_dim ∈ {64, 128} |

## 进入下一关

通过后进入 [L08.5 quant serving](../l23_quant_serving/README.md)。
