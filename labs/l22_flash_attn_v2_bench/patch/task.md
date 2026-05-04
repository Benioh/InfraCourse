# L08.3 Patch · FlashAttention v2 benchmark

## 你要交付什么

```python
def eager_attention(q, k, v, causal: bool = False) -> torch.Tensor:
    """[B, H, T, D] in / [B, H, T, D] out, manual softmax(QK^T / sqrt(D)) V."""

def flash_attention(q, k, v, causal: bool = False) -> torch.Tensor:
    """Use torch.nn.functional.scaled_dot_product_attention with the FA backend."""

def bench_attention(
    seq_len: int, num_heads: int, head_dim: int, causal: bool,
    device: str, dtype: torch.dtype, num_iters: int = 20,
) -> dict:
    """Return {"eager_time_ms", "flash_time_ms", "speedup",
              "max_abs_diff", "peak_mem_eager_mb", "peak_mem_flash_mb"}.
    Numerical comparison done in fp32; perf in the requested dtype."""
```

补丁规模目标：60-100 行。

## 不变量

1. `eager_attention` 必须显式实例化 N×N attention matrix（这是 baseline）
2. `flash_attention` 必须调 `torch.nn.functional.scaled_dot_product_attention`
3. CPU / fp32 时两者输出 `max_abs_diff < 1e-5`
4. `bench_attention` 必须 `torch.cuda.synchronize()` 包裹计时
5. `peak_mem_*_mb` 用 `torch.cuda.max_memory_allocated()` 测；CPU 时填 0

## 怎么验证

```bash
make patch-test M=l22_flash_attn_v2_bench
RUN_GPU_TESTS=1 make patch-test M=l22_flash_attn_v2_bench   # GPU 真实 speedup
```

## 写完之后你能做什么

- 在自己的 transformer 里替换 eager attention 为 SDPA / FA
- 解释 vLLM PagedAttention 为什么会受 FA dispatch 影响
- 知道 `torch.backends.cuda.sdp_kernel` 怎么强制选 backend
