"""Reference solution for L01.7 Patch · Triton softmax."""

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
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(
            input_ptr + row_idx * input_row_stride + col_offsets,
            mask=mask,
            other=-float("inf"),
        )
        row_max = tl.max(row, axis=0)
        numer = tl.exp(row - row_max)
        denom = tl.sum(numer, axis=0)
        out = numer / denom
        tl.store(
            output_ptr + row_idx * output_row_stride + col_offsets,
            out,
            mask=mask,
        )


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("triton not installed")
    if not x.is_cuda:
        raise RuntimeError("triton_softmax requires CUDA tensor")
    assert x.dim() == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
