"""Fused BF16 bias add (fouroversix) and gated residual must match eager bit for bit (GPU).

python tests/test_fused_elementwise.py   (run from LongLive/)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fouroversix.model.modules.bias_add import bias_add_, can_fuse_bias_add
from utils.residual_triton import can_fuse_gated_residual, gated_residual

torch.manual_seed(0)


def test_bias_add_bitexact():
    with torch.no_grad():
        for n in (3072, 14336, 3000):
            out = (torch.randn(7040, n, device="cuda") * 3).bfloat16()
            bias = torch.randn(n, device="cuda").bfloat16()
            want = out + bias
            assert can_fuse_bias_add(out, bias)
            assert torch.equal(bias_add_(out.clone(), bias), want), n


def test_gated_residual_bitexact():
    with torch.no_grad():
        B, F, S, C = 1, 8, 880, 3072
        x = (torch.randn(B, F * S, C, device="cuda") * 4).bfloat16()
        y = torch.randn(B, F * S, C, device="cuda").bfloat16()
        e = torch.randn(B, F, 6, C, device="cuda").bfloat16().chunk(6, dim=2)   # the model's gate views
        for gate in (e[2], e[5]):
            want = x + (y.unflatten(dim=1, sizes=(F, S)) * gate).flatten(1, 2)
            assert can_fuse_gated_residual(x, y, gate)
            assert torch.equal(gated_residual(x, y, gate, S), want)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
