"""Bit-exact fused `WanRMSNorm.forward` for BF16 CUDA inputs.

Keeps PyTorch's own mean reduction (same summation order) and fuses the ops around it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _square_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, x * x, mask=mask)


@triton.jit
def _scale_kernel(x_ptr, mean_ptr, w_ptr, out_ptr, C, eps, BLOCK_C: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    # eager: `mean + eps` then `torch.rsqrt` (CUDA rsqrtf == libdevice rsqrt)
    r = libdevice.rsqrt(tl.load(mean_ptr + row) + eps)
    x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
    normed = (x * r).to(tl.bfloat16)                                   # .type_as(x)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)  # bf16 * bf16 in fp32 opmath
    tl.store(out_ptr + row * C + offs, out, mask=mask)


def rmsnorm_bitexact(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`WanRMSNorm.forward(x)` for contiguous BF16 CUDA `x` and BF16 `weight`."""
    C = x.shape[-1]
    x_c = x.contiguous()
    sq = torch.empty(x_c.shape, dtype=torch.float32, device=x.device)
    n = x_c.numel()
    _square_kernel[(triton.cdiv(n, 4096),)](x_c, sq, n, BLOCK=4096, num_warps=8)
    mean = sq.mean(dim=-1, keepdim=True)
    out = torch.empty_like(x_c)
    _scale_kernel[(n // C,)](x_c, mean, weight, out, C, eps, BLOCK_C=triton.next_power_of_2(C), num_warps=8)
    return out
