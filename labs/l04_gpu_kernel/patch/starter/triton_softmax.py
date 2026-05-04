"""
L01.7 Patch · Triton 行 softmax

填空规则：
- TODO(student) 必须自己写
- 不许用 F.softmax / torch.softmax
- 允许 triton.* / triton.language.*

完成度自检：
    make patch-test M=l04_gpu_kernel
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def softmax_kernel(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        """每个 program 处理一行：load → max → exp → sum → div → store."""
        # TODO(student):
        #   1. row_idx = tl.program_id(0)
        #   2. col_offsets = tl.arange(0, BLOCK_SIZE)
        #   3. mask = col_offsets < n_cols
        #   4. row = tl.load(input_ptr + row_idx * input_row_stride + col_offsets,
        #                    mask=mask, other=-float("inf"))
        #      用 -inf 让 mask 位置在 max 时被忽略。
        #   5. row_max = tl.max(row, axis=0)
        #   6. numer = tl.exp(row - row_max)
        #   7. denom = tl.sum(numer, axis=0)
        #   8. out = numer / denom
        #   9. tl.store(output_ptr + row_idx * output_row_stride + col_offsets, out, mask=mask)
        pass


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("triton not installed; install pip install triton")
    if not x.is_cuda:
        raise RuntimeError("triton_softmax requires CUDA tensor")
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    # TODO(student):
    #   1. n_rows, n_cols = x.shape
    #   2. BLOCK_SIZE = triton.next_power_of_2(n_cols)
    #      （Triton 的 BLOCK_SIZE 必须是 2 的幂）
    #   3. output = torch.empty_like(x)
    #   4. softmax_kernel[(n_rows,)](
    #        output, x,
    #        x.stride(0), output.stride(0),
    #        n_cols,
    #        BLOCK_SIZE=BLOCK_SIZE,
    #      )
    #   5. return output
    raise NotImplementedError("L01.7 Patch: implement triton_softmax")
