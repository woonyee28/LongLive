"""utils.rmsnorm_triton must match the eager WanRMSNorm bit for bit (needs a GPU).

python tests/test_rmsnorm_triton.py   (run from LongLive/)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rmsnorm_triton import rmsnorm_bitexact
from wan_5b.modules.model import WanRMSNorm


def eager(norm, x):
    return norm._norm(x.float()).type_as(x) * norm.weight


def check(shape, scale, seed):
    torch.manual_seed(seed)
    norm = WanRMSNorm(shape[-1], eps=1e-6).cuda().bfloat16()
    norm.weight.data = (1 + 0.3 * torch.randn_like(norm.weight.float())).bfloat16()
    x = (torch.randn(shape, device="cuda") * scale).bfloat16()
    want, got = eager(norm, x), rmsnorm_bitexact(x, norm.weight, norm.eps)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want), f"{shape} x{scale}: {(got != want).sum().item()} elements differ"


def test_bitexact_attention_shapes():
    for i, scale in enumerate((0.02, 1.0, 30.0)):
        check((1, 7040, 3072), scale, i)          # self-attention q/k of one block


def test_bitexact_cross_attention_shape():
    check((1, 512, 3072), 1.0, 7)                 # cross-attention k over the text tokens


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
