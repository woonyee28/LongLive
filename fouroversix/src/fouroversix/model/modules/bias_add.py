"""Vectorised in-place BF16 bias add, bit-identical to `out + bias`."""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


_ENABLED = os.environ.get("LLV2_TRITON_BIAS_ADD", "1") == "1"


@triton.jit
def _bias_add_kernel(out_ptr, bias_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]
    x = tl.load(out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (x + b[None, :]).to(tl.bfloat16), mask=mask)


def can_fuse_bias_add(out: torch.Tensor, bias: torch.Tensor) -> bool:
    return (
        _ENABLED
        and not torch.is_grad_enabled()
        and out.is_cuda
        and out.dtype == torch.bfloat16
        and bias.dtype == torch.bfloat16
        and out.is_contiguous()
        and bias.is_contiguous()
        and bias.dim() == 1
        and bias.shape[0] == out.shape[-1]
    )


def bias_add_(out: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """In-place `out += bias` over the last dim; same values as `out + bias`."""
    N = out.shape[-1]
    M = out.numel() // N
    BLOCK_M, BLOCK_N = 32, 256
    _bias_add_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        out, bias, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8,
    )
    return out
