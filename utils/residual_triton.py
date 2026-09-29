"""Bit-exact fused adaLN gated residual `x + y * gate` for BF16 inference."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_residual_kernel(
    x_ptr, y_ptr, g_ptr, out_ptr,
    L, C, FRAME_SEQLEN,
    g_stride_b, g_stride_f,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)                  # over B * L tokens
    b = row // L
    f = (row % L) // FRAME_SEQLEN
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < C
    x = tl.load(x_ptr + row * C + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * C + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + b * g_stride_b + f * g_stride_f + cols, mask=mask, other=0.0).to(tl.float32)
    gated = (y * g).to(tl.bfloat16).to(tl.float32)     # eager: y * gate -> bf16
    tl.store(out_ptr + row * C + cols, (x + gated).to(tl.bfloat16), mask=mask)


def can_fuse_gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> bool:
    return (
        not torch.is_grad_enabled()
        and x.is_cuda
        and x.dtype == y.dtype == gate.dtype == torch.bfloat16
        and x.dim() == 3 and x.shape == y.shape
        and x.is_contiguous() and y.is_contiguous()
        and gate.dim() == 4 and gate.shape[2] == 1 and gate.shape[3] == x.shape[2]
        and gate.stride(3) == 1
    )


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor, frame_seqlen: int) -> torch.Tensor:
    """`x + (y.unflatten(1, (F, frame_seqlen)) * gate).flatten(1, 2)`; gate is [B, F, 1, C]."""
    B, L, C = x.shape
    out = torch.empty_like(x)
    BLOCK_C = 1024
    _gated_residual_kernel[(B * L, triton.cdiv(C, BLOCK_C))](
        x, y, gate, out, L, C, frame_seqlen,
        gate.stride(0), gate.stride(1),
        BLOCK_C=BLOCK_C, num_warps=4,
        enable_fp_fusion=False,   # an FMA would skip the bf16 rounding of y * gate
    )
    return out
